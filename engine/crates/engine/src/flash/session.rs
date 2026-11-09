//! Batch-one text sessions with resident or layer-streamed weights and caches.
//! Execute one distributed rank or two local ranks sharing one GPU sequentially.
//! Official weights are never mirrored on the host.
use super::{Weights, blocks::{Hc, HcScratch, Ple, PLE_TOKENS, H, W},
    rows::{Rows, RowBatch, FrontScratch, EMBEDDING_BYTES, LEXICAL_BYTES}, gdn::{Gdn, GdnState, GdnScratch, GdnBatch},
    qsa::{Qsa, QsaState, QsaScratch, QsaGather, QsaBatch}, moe::{Moe, MoeScratch, MoePartials, PARTIALS},
    head::{Head, Candidate, VOCABULARY}};
use engine_model::flash::{Checkpoint, LayerKind, partition, plan::{RankPlan, PREFILL_PROFILE_CLASSES}};
use spark_gpu_probe::{driver::{Backend, DecodeGraph, DeviceVector, HostRegistration, ExchangeSlot, Fence, Fp8RoutingFlag, PromptBuffer}, kernel_params::HybridCopy};
use std::{path::{Path, PathBuf}, time::Instant};
mod cache;
pub(super) mod mtp;
mod prefill;
mod forward;
use mtp::{Draft, Verification};
pub use cache::Snapshot;
use cache::Recurrent;
use crate::flash_candidate::{RoutingVerdict, NONFINITE_LOGIT, ROUTING_ERROR};
use std::{cell::RefCell, rc::Rc};

type Result<T> = std::result::Result<T, String>;
pub enum SampleRow { Logits(Vec<f32>), TopK(Vec<(u32, f32)>) }
const PLE: &str = "model.language_model.layers.1.ple.";
const MIXER: &str = "model.language_model.hyper_connection_mixer";
pub const SHARED_PREFIX_TOKENS: usize = 5312;

pub use engine_fabric::exchange::{Exchange, ExchangeStats, Profiled};
fn parallel<E: Exchange, T: Send>(ranks: &mut [E], values: Vec<T>,
    f: impl Fn(&mut E, T) -> Result<T> + Sync) -> Result<Vec<T>> {
    // A distributed rank has one local transport: no worker thread per collective.
    if ranks.len() == 1 {
        return values.into_iter().map(|value| f(&mut ranks[0], value)).collect();
    }
    std::thread::scope(|s| {
        let jobs = ranks.iter_mut().zip(values).map(|(rank, value)| {
            let f = &f;
            s.spawn(move || f(rank, value))
        }).collect::<Vec<_>>();
        jobs.into_iter().map(|j| j.join().map_err(|_| "Flash collective worker panicked")?)
            .collect()
    })
}

/// Largest single collective: one 64-token MoE partial tile (the RDMA mailbox
/// slot). Longer vectors reduce piecewise; every element keeps its rank-ordered sum.
const REDUCE_PIECE: usize = 64 * 11 * H;
/// Tokens that pass through every layer together. Attention still runs in
/// 64-token chunks at the same prompt offsets; the per-token HC, router, shared
/// and routed experts then read their weights once per group instead of per chunk.
fn group_tokens(capacity: usize, tile: usize) -> usize {
    if tile == 64 { capacity.min(4096).div_ceil(64) * 64 } else { tile }
}
/// Device bytes per group token. Computed: session input and hidden rows
/// 51,200 + two HC tile workspaces 2 x 83,216 + the MoE arena 140,492 + ten
/// W8A8 grouped rows 10 x 16,020 with token operands and routing words 2,765
/// = 521,089. The LONG ledger measured ~547 KB per token; that is reserved.
const GROUP_TOKEN_BYTES: usize = 547_004;
/// The W8A8 grouped workspace pads each routed expert by up to 15 rows of 16,020 bytes.
const EXPERT_PADDING_BYTES: usize = 15 * 512 * 16_020;
fn group_bytes(tokens: usize) -> usize {
    tokens * GROUP_TOKEN_BYTES
        - tokens.saturating_sub(HcScratch::workspace_tokens()) * 2 * HcScratch::WORKSPACE_TOKEN_BYTES
        + 15 * (tokens * 10).min(512) * 16_020
}
fn prompt_bytes(plan: &RankPlan, group: usize, slots: usize) -> usize {
    let heads = (plan.value_heads.end - plan.value_heads.start) as usize;
    let keys = (plan.key_heads.end - plan.key_heads.start) as usize;
    slots * (group_bytes(group) + if plan.expert_output_rows.is_some() { compact_bytes(plan.ranks as usize, group) } else { 0 })
        + GdnScratch::group_reservation_bytes(heads, keys, group.min(GDN_SPAN))
}
fn compact_bytes(world: usize, tokens: usize) -> usize {
    if world == 2 { 0 } else {
        (world + 1) * (tokens * spark_gpu_probe::driver::fp8_abi::middle_packet_words(world) * 4).div_ceil(256) * 256
    }
}
/// Groups a resident session carries through one layer loop together. The host
/// finishes one group's collective while the GPU runs the next group's queued
/// segments; every group's own sums keep their piece and rank order.
const GROUPS_IN_FLIGHT: usize = 4;
fn group_slots(resident: bool, mailbox: bool, compact: bool) -> usize {
    // Output-owned expert gathers finish synchronously, so extra groups do
    // not overlap their exchanges. Reuse one group's scratch on this path;
    // keep the same 4096-token group, 64-token attention tiles and rank order.
    if resident && !mailbox && !compact { GROUPS_IN_FLIGHT } else { 1 }
}

/// GDN verification slices input rows inside graph capture; MTP retains hidden
/// row views afterwards. Initial, prompt and resized execution buffers must
/// share this arena ownership contract, including the late sole-owner claim.
fn execution_rows<'a>(b: &Backend<'a>, group: usize) -> Result<(DeviceVector<'a>, DeviceVector<'a>)> {
    let input = b.vector_arena(&[group * H])?.remove(0);
    let hidden = b.vector_arena(&[group * W])?.remove(0);
    Ok((input, hidden))
}

/// Prompt groups are serial on the model's ordered stream. Keep their large
/// workspaces once per model; sessions retain only a tile for decode and MTP.
struct PromptScratch<'a> {
    front: Vec<FrontScratch<'a>>,
    /// One recurrence span per rank, reused by groups in stream order.
    gdn: Vec<GdnScratch<'a>>,
    hc: Vec<HcScratch<'a>>,
    moe: Vec<MoeScratch<'a>>,
    input: Vec<DeviceVector<'a>>,
    hidden: Vec<DeviceVector<'a>>,
    exchange: Vec<ExchangeSlot<'a>>,
    staging: Vec<PromptBuffer<'a>>,
    group: usize,
    slots: usize,
    mailbox: bool,
    routing: Vec<Fp8RoutingFlag<'a>>,
}
struct PreparedRows {
    batches: Vec<Vec<Vec<RowBatch>>>,
    after: Vec<[i64; 2]>,
    boundary: Option<Vec<[i64; 2]>>,
}
struct PendingRows {
    prepared: PreparedRows,
    offsets: Vec<(usize, usize, usize, usize)>,
    ranks: Vec<usize>,
    group: usize,
    tile: usize,
}
impl PendingRows {
    fn start_pack(mut self, b: &Backend<'_>, slot: &mut PromptBuffer<'_>,
        send: std::sync::mpsc::Sender<PreparedRows>) -> Result<()> {
        b.profile_host_pack(|| slot.start_pack(b.is_profiling(), move |bytes| {
            // Finish all reads before reusing raw-wire space for the planes.
            for (g, r, index, offset) in self.offsets {
                self.prepared.batches[g][r][index].fill_prompt(bytes, offset, self.ranks[r]);
            }
            for (g, ranks) in self.prepared.batches.iter().enumerate() {
                for (r, batches) in ranks.iter().enumerate() {
                    let base = (g * self.ranks.len() + r) * self.group * (EMBEDDING_BYTES + LEXICAL_BYTES);
                    for (index, batch) in batches.iter().enumerate() {
                        batch.pack_prompt(bytes, base, index * self.tile, self.group, self.ranks[r], true);
                    }
                }
            }
            let _ = send.send(self.prepared);
            Ok(())
        }))
    }
}
/// Sixteen canonical 64-token chunks per prompt recurrence launch.
const GDN_SPAN: usize = 1024;
fn recurrence_spans(count: usize, boundary: usize) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    while start < count {
        let end = (start + GDN_SPAN).min(count);
        let end = if start < boundary && boundary < end { boundary } else { end };
        spans.push((start, end - start));
        start = end;
    }
    spans
}
#[test]
fn recurrence_spans_keep_chunks_and_checkpoint() {
    for (group, bytes) in [(64, 13_669_440), (1024, 218_697_600), (4096, 218_697_600)] { for slots in [1, 4] { assert_eq!(prompt_bytes(&RankPlan::new(0, 1).unwrap(), group, slots) - slots * group_bytes(group), bytes); } }
    for count in [65, 257, 1025, 4095, 4096, 8192] { for boundary in [0, 64, 256, 960, 1024, 4032, 4096] {
        let spans = recurrence_spans(count, boundary);
        assert!(spans.iter().all(|&(s, n)| n <= GDN_SPAN && !(s < boundary && boundary < s + n)));
        assert_eq!(spans.iter().flat_map(|&(s, n)| (s..s + n).step_by(64)).collect::<Vec<_>>(), (0..count).step_by(64).collect::<Vec<_>>());
    } }
}
// Measure the complete exchange point, including staging before the trait
// call and queued uploads afterwards. Transport/Flash/point spans are nested.
fn exchange_point<E: Exchange, T>(ranks: &mut [E], kind: usize,
    work: impl FnOnce(&mut [E]) -> Result<T>) -> Result<T> {
    if !ranks.first().is_some_and(Exchange::collective_profiling) { return work(ranks); }
    let start = Instant::now();
    let result = work(ranks);
    let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    for rank in ranks { rank.profile_point(kind, ns, result.is_ok()); }
    result
}
fn reduce_pieces<E: Exchange>(rank: &mut E, values: &mut [f32], scratch: &mut [f32]) -> Result<()> {
    values.chunks_mut(REDUCE_PIECE).try_for_each(|piece| rank.reduce(piece, scratch))
}

/// Queue the peer's FP32 values in a receive slot into device values at `own`.
fn add_peer(b: &Backend<'_>, mailbox: &HostRegistration<'_>, own: u64, peer: &[u8]) -> Result<()> {
    let n = peer.len() / 4;
    super::blocks::add_rows(b, own, mailbox.device_address(peer.as_ptr())?, n, 1, 1)
}
/// A failed later launch can leave an earlier consumer queued without an event.
/// Drain it before returning the error to the owner of the registered slot.
fn queue_consumer<'a>(b: &Backend<'a>, enqueue: impl FnOnce() -> Result<()>) -> Result<Fence<'a>> {
    if let Err(error) = enqueue() { b.synchronize()?; return Err(error); }
    b.fence()
}

/// A prompt MLP of a distributed rank, reduced by token halves: rank 0 owns
/// the first ceil(count/2) tokens, rank 1 the rest. The peer sends the partial
/// rows of this rank's tokens (64-token pieces) and they are added in place
/// (rank 0 + rank 1, as reduce_device); this rank finishes its tokens, then the
/// finished output rows are swapped. Every token's finish reads the same
/// reduced rows and routing on either rank, so outputs equal the full
/// reduction's; about half the bytes cross the link.
fn reduce_finish_split<E: Exchange>(b: &Backend<'_>, rank: &mut E, mailbox: &HostRegistration<'_>,
    s: &mut MoeScratch<'_>, mlp: &Moe<'_>, first: bool, count: usize) -> Result<()> {
    b.prefill_collective(|| reduce_finish_split_inner(b, rank, mailbox, s, mlp, first, count))
}
fn reduce_finish_split_inner<E: Exchange>(b: &Backend<'_>, rank: &mut E, mailbox: &HostRegistration<'_>,
    s: &mut MoeScratch<'_>, mlp: &Moe<'_>, first: bool, count: usize) -> Result<()> {
    let half = count.div_ceil(2);
    let (own, theirs) = if first { (0..half, half..count) } else { (half..count, 0..half) };
    let piece = |range: &std::ops::Range<usize>, p: usize| (range.start + p * 64).min(range.end)..(range.start + p * 64 + 64).min(range.end);
    let values = MoePartials::VALUES;
    let pieces = (0..half.div_ceil(64)).map(|p| (piece(&theirs, p), piece(&own, p))).collect::<Vec<_>>();
    let sizes = pieces.iter().map(|(s, r)| (s.len() * values * 4, r.len() * values * 4)).collect::<Vec<_>>();
    b.prefill_read_bytes(9, || sizes.iter().map(|(s, r)| s + r).sum());
    let partials = s.partials();
    rank.exchange_slots_pipeline(&sizes,
        |p, slot| unsafe { b.download_range(partials, pieces[p].0.start * values, slot) },
        |p, peer| queue_consumer(b, || {
            let rows = &pieces[p].1;
            add_peer(b, mailbox, partials.device_address() + (rows.start * values * 4) as u64, peer)?;
            // Each token keeps its expert/rank sum order. Finish these rows
            // while the next piece crosses the NIC; no second workspace.
            mlp.finish_rows(b, s, rows.start, rows.len())
        }), |event| b.wait_fence(event))?;
    let output = s.output();
    // Finished rows also obey the existing mailbox bound when prompt groups
    // exceed 1024 tokens. Each piece is at most 5 MiB; rank/token order stays fixed.
    let pieces = (0..half).step_by(512).map(|start| {
        let slice = |r: &std::ops::Range<usize>| (r.start + start).min(r.end)..(r.start + start + 512).min(r.end);
        (slice(&own), slice(&theirs))
    }).collect::<Vec<_>>();
    let sizes = pieces.iter().map(|(s, r)| (s.len() * H * 4, r.len() * H * 4)).collect::<Vec<_>>();
    b.prefill_read_bytes(9, || sizes.iter().map(|(s, r)| s + r).sum());
    rank.exchange_slots_pipeline(&sizes,
        |p, slot| unsafe { b.download_range(output, pieces[p].0.start * H, slot) },
        |p, peer| queue_consumer(b, || {
            let at = output.device_address() + (pieces[p].1.start * H * 4) as u64;
            let words = peer.len() / 4;
            let mut copy = HybridCopy { source: mailbox.device_address(peer.as_ptr())? as *const u32,
                destination: at as *mut u32, words: words as u32, padding: 0, rows: 0 };
            unsafe { b.launch_flash_rows("flash_hybrid_copy", words.div_ceil(128) as u32, 1, 128, 0,
                &mut [(&mut copy as *mut HybridCopy).cast()]) }
        }), |event| b.wait_fence(event))?;
    Ok(())
}

/// Two true gathers around output-owned expert projections. The same byte
/// copies serve prompt tiles, graph-backed verification and independent chats.
fn exchange_middle<E: Exchange>(b: &Backend<'_>, rank: &mut E, mailbox: Option<&HostRegistration<'_>>,
    s: &MoeScratch<'_>, own: usize, tokens: usize, finished: bool, wire: &mut [f32]) -> Result<()> {
    b.prefill_read_bytes(9, || { let world = s.middle_world().unwrap_or(2);
        tokens * 4 * if finished { H + spark_gpu_probe::driver::fp8_abi::middle_columns(own, world).len() * world.saturating_sub(2) }
        else { spark_gpu_probe::driver::fp8_abi::middle_packet_words(world) * world.saturating_sub(1) * 2 } });
    b.prefill_collective(|| exchange_point(std::slice::from_mut(rank), 1, |ranks|
        exchange_middle_inner(b, &mut ranks[0], mailbox, s, own, tokens, finished, wire)))
}
fn exchange_middle_inner<E: Exchange>(b: &Backend<'_>, rank: &mut E, mailbox: Option<&HostRegistration<'_>>,
    s: &MoeScratch<'_>, own: usize, tokens: usize, finished: bool, wire: &mut [f32]) -> Result<()> {
    let world = s.middle_world()?;
    if mailbox.is_some() && s.small_middle(tokens) {
        return s.exchange_small_middle(b, rank, own, tokens, finished);
    }
    if world != 2 {
        let (send, mut receive) = s.middle_packets()?;
        let words = spark_gpu_probe::driver::fp8_abi::middle_packet_words(world);
        let span = send.len().div_ceil(64) * 64;
        let widths = (0..world).map(|r| if finished {
            spark_gpu_probe::driver::fp8_abi::middle_columns(r, world).len()
        } else { words }).collect::<Vec<_>>();
        let piece = (REDUCE_PIECE / words).min(512).min(wire.len() / world / words);
        if own >= world || piece == 0 { return Err("compact gather scratch capacity".into()); }
        if let Some(mailbox) = mailbox {
            // Retain the local packet in its canonical rank slice, entirely on device.
            b.copy_vector_range(&send, &mut receive, 0, own * span, tokens * widths[own])?;
            let offsets = (0..tokens).step_by(piece).map(|start| {
                let mut offsets = vec![0];
                for width in &widths { offsets.push(offsets.last().unwrap() + (tokens - start).min(piece) * width * 4); }
                offsets
            }).collect::<Vec<_>>();
            return rank.gather_slots_pipeline(&offsets,
                |p, at, slot| unsafe { b.download_range(&send, p * piece * widths[own] + at / 4, slot) },
                |p, r, at, peer| queue_consumer(b, || {
                    let mut copy = HybridCopy { source: mailbox.device_address(peer.as_ptr())? as *const u32,
                        destination: (receive.device_address() + ((r * span + p * piece * widths[r]) * 4 + at) as u64) as *mut u32,
                        words: (peer.len() / 4) as u32, padding: 0, rows: 0 };
                    unsafe { b.launch_flash_rows("flash_hybrid_copy", copy.words.div_ceil(128), 1, 128, 0,
                        &mut [(&mut copy as *mut HybridCopy).cast()]) }
                }), |event| b.wait_fence(event));
        }
        for start in (0..tokens).step_by(piece) {
            let count = (tokens - start).min(piece);
            let mut offsets = vec![0];
            for width in &widths { offsets.push(offsets.last().unwrap() + count * width); }
            let wire = &mut wire[..offsets[world]];
            b.read_vector_range_into(&send, start * widths[own], &mut wire[offsets[own]..offsets[own + 1]])?;
            // Opaque byte gather: NaN-shaped FP8/scale words never enter arithmetic.
            let bytes = unsafe { std::slice::from_raw_parts_mut(wire.as_mut_ptr().cast::<u8>(), wire.len() * 4) };
            rank.gather_bytes(bytes, &offsets.iter().map(|n| n * 4).collect::<Vec<_>>())?;
            for r in 0..world {
                if widths[r] != 0 {
                    b.write_vector_range(&mut receive, r * span + start * widths[r], &wire[offsets[r]..offsets[r + 1]])?;
                }
            }
        }
        return Ok(());
    }
    let (send, receive) = s.middle_packets()?;
    let words = if finished { tokens * H / 2 } else { send.len() };
    let send = send.slice(0..words)?;
    let mut receive = receive.slice(0..words)?;
    if let Some(mailbox) = mailbox {
        let sizes = (0..words).step_by(REDUCE_PIECE)
            .map(|at| { let bytes = (words - at).min(REDUCE_PIECE) * 4; (bytes, bytes) }).collect::<Vec<_>>();
        rank.exchange_slots_pipeline(&sizes,
            |p, slot| unsafe { b.download_range(&send, p * REDUCE_PIECE, slot) },
            |p, peer| queue_consumer(b, || {
                let mut copy = HybridCopy { source: mailbox.device_address(peer.as_ptr())? as *const u32,
                    destination: (receive.device_address() + (p * REDUCE_PIECE * 4) as u64) as *mut u32,
                    words: (peer.len() / 4) as u32, padding: 0, rows: 0 };
                unsafe { b.launch_flash_rows("flash_hybrid_copy", copy.words.div_ceil(128), 1, 128, 0,
                    &mut [(&mut copy as *mut HybridCopy).cast()]) }
            }), |event| b.wait_fence(event))
    } else {
        // The ordinary gather copies opaque words; no FP32 arithmetic touches
        // the E4M3 codes or scale bits in this transport fallback.
        let mut wire = vec![0.0; words * 2];
        wire[own * words..(own + 1) * words].copy_from_slice(&b.read_vector_values(&send)?);
        rank.gather_f32(&mut wire, &[0, words, 2 * words])?;
        b.write_vector(&mut receive, &wire[(1 - own) * words..(2 - own) * words])
    }
}

fn reduce_device<E: Exchange>(b: &Backend<'_>, ranks: &mut [E], scratch: &mut [Vec<f32>],
    mailbox: Option<&HostRegistration<'_>>, values: &mut [&mut DeviceVector<'_>]) -> Result<()> {
    b.prefill_read_bytes(9, || values.iter().map(|v| v.len() * 8).sum());
    b.prefill_collective(|| exchange_point(ranks, 0, |ranks| reduce_device_inner(b, ranks, scratch, mailbox, values)))
}
fn reduce_device_inner<E: Exchange>(b: &Backend<'_>, ranks: &mut [E], scratch: &mut [Vec<f32>],
    mailbox: Option<&HostRegistration<'_>>, values: &mut [&mut DeviceVector<'_>]) -> Result<()> {
    if let (1, Some(mailbox)) = (ranks.len(), mailbox) {
        // Prompts: the copy engine fills the send slot; flash_add adds the peer's
        // slot in place: rank 0 + rank 1 for every element (FP32 addition
        // commutes), as the host sum gives, without host copies or sums. Decode
        // keeps the host path, whose upload needs no wait before the next graph.
        let values = &mut *values[0];
        let sizes = (0..values.len()).step_by(REDUCE_PIECE).map(|start| {
            let bytes = REDUCE_PIECE.min(values.len() - start) * 4;
            (bytes, bytes)
        }).collect::<Vec<_>>();
        ranks[0].exchange_slots_pipeline(&sizes,
            |p, send| unsafe { b.download_range(values, p * REDUCE_PIECE, send) },
            |p, peer| queue_consumer(b, ||
                add_peer(b, mailbox, values.device_address() + (p * REDUCE_PIECE * 4) as u64, peer)),
            |event| b.wait_fence(event))?;
        return Ok(());
    }
    if ranks.len() == 1 {
        return b.exchange_vector(values[0], |v| reduce_pieces(&mut ranks[0], v, &mut scratch[0]));
    }
    // The two-local-rank debugging path must stage both ranks before either
    // blocks in the collective. Distributed execution uses the pinned slot.
    let mut host = values.iter().map(|v| b.read_vector_values(v)).collect::<Result<Vec<_>>>()?;
    std::thread::scope(|s| {
        let jobs = ranks.iter_mut().zip(&mut host).zip(scratch)
            .map(|((rank, values), scratch)| s.spawn(move || reduce_pieces(rank, values, scratch)))
            .collect::<Vec<_>>();
        for job in jobs { job.join().map_err(|_| "Flash reduction worker panicked")??; }
        Ok::<(), String>(())
    })?;
    for (device, host) in values.iter_mut().zip(host) { b.write_vector(device, &host)?; }
    Ok(())
}

pub struct Model<'a> {
    root: PathBuf,
    checkpoint: Checkpoint,
    /// LBF12 twins (HC up, attention): declared before the weights they mirror, so they drop first.
    _lbf12: super::blocks::Lbf12Twins<'a>, persistent: Vec<Weights<'a>>,
    resident: bool, activation_quantization: bool,
    mtp_depth: usize,
    profile_layers: bool,
    prompt: RefCell<Option<PromptScratch<'a>>>,
    /// The single rank's draft-only MXFP8 head image (Head::fp8_draft_twin), if enabled.
    draft_mx8: Option<DeviceVector<'a>>,
}
impl<'a> Model<'a> {
    pub fn is_resident(&self) -> bool { self.resident }
    pub fn layer_profiling(&self) -> bool { self.profile_layers }
    pub fn mtp_depth(&self) -> usize { self.mtp_depth }
    pub fn resident_bytes(&self) -> usize {
        self.persistent.iter().map(Weights::resident_bytes).sum()
    }
    /// Serving keeps a plain decode row per session. The sole draft owner's
    /// wider scratch and history are funded once outside the admission pool.
    pub fn session_reservation_bytes(&self, capacity: usize, tile: usize, mailbox: bool) -> usize {
        let slots = self.workspace_slots(mailbox);
        self.persistent.iter().map(|weights| session_bytes(&weights.plan, capacity, tile, slots, 1)).sum::<usize>()
            + self.graph_session_reservation_bytes()
    }
    /// Each live chat may retain its one-row target graphs.
    pub fn graph_session_reservation_bytes(&self) -> usize {
        self.graph_reservation_bytes_for_depth(0)
    }
    /// Verification graphs belong to the sole draft owner, never every chat.
    pub fn graph_draft_reservation_bytes(&self) -> usize {
        self.graph_reservation_bytes_for_depth(self.mtp_depth) - self.graph_session_reservation_bytes()
    }
    fn graph_reservation_bytes_for_depth(&self, depth: usize) -> usize {
        if !self.resident || self.persistent.len() != 1 { return 0; }
        let segments = if self.persistent[0].plan.expert_output_rows.is_some() { 3 } else { 2 };
        (self.checkpoint.config.layers.len() * segments + 24) * (1..=depth + 1)
            .map(DecodeGraph::reservation_bytes_for_rows).sum::<usize>()
            + if depth == 0 { 0 } else { 3 * DecodeGraph::reservation_bytes_for_rows(1) }
    }
    /// Fund a full-context draft owner once, including history, wider target
    /// scratch and old/new growth. Driver graphs have their own shared charge.
    pub fn draft_reservation_bytes(&self, capacity: usize) -> Result<usize> {
        if self.mtp_depth == 0 { return Ok(0); }
        let world = self.persistent[0].plan.ranks;
        let workspace = (0..world).try_fold(0, |maximum, rank| -> Result<usize> {
            let plan = RankPlan::new(rank, world).map_err(|e| e.to_string())?;
            Ok(maximum.max(mtp::reservation_bytes(&plan, capacity, self.mtp_depth)))
        })?;
        // Cached publication can mask part of execution growth, so cover both
        // cache policies when funding the full-context owner's request delta.
        let slots = self.workspace_slots(false);
        let mut delta = 0;
        for caching in [false, true] {
            let plain = self.request_reservation_bytes(capacity, 64, slots, caching)?;
            let draft = self.request_reservation_bytes_for_depth(capacity, 64, slots, caching, self.mtp_depth)?;
            delta = delta.max(draft - plain);
        }
        workspace.checked_add(delta - self.graph_draft_reservation_bytes())
            .ok_or_else(|| "Flash draft reservation overflow".into())
    }
    pub fn workspace_slots(&self, _mailbox: bool) -> usize {
        1
    }
    /// Main NVFP4 experts gather BF16 operands; optional MTP remains FP8.
    pub fn main_activation_quantization(&self) -> bool {
        self.checkpoint.config.format != engine_model::flash::WeightFormat::Nvfp4 && self.activation_quantization
    }
    pub fn expert_precision(&self) -> &'static str {
        if self.checkpoint.config.format == engine_model::flash::WeightFormat::Nvfp4 {
            "w4a16_e2m1_k16_scales_bf16_operands_fp32_accumulation"
        } else { Moe::precision_name(self.activation_quantization) }
    }
    /// An idle prompt fills the existing workspaces so host collectives can
    /// overlap the next group's GPU work. Contended serving still uses a tile.
    pub fn prefill_chunk_size(&self, capacity: usize, tile: usize, mailbox: bool) -> usize {
        group_tokens(capacity, tile) * group_slots(self.resident, mailbox,
            self.persistent[0].plan.expert_output_rows.is_some())
    }
    /// Allocate shared prompt scratch once, before serving admission. Return
    /// the remaining conservative reservation beyond its physical allocations.
    pub fn prepare_prompt(&self, b: &Backend<'a>, capacity: usize, tile: usize, mailbox: bool) -> Result<usize> {
        let mailbox = mailbox && self.persistent[0].plan.is_pair();
        b.prepare_flash_dense()?;
        let group = group_tokens(capacity, tile);
        let slots = group_slots(self.resident, mailbox, self.persistent[0].plan.expert_output_rows.is_some());
        let mut prompt = self.prompt.try_borrow_mut().map_err(|_| "Flash prompt workspace is in use")?;
        if prompt.as_ref().is_none_or(|p| p.group < group || p.slots != slots || p.mailbox != mailbox) {
            let group = prompt.as_ref().map_or(group, |p| p.group.max(group));
            b.synchronize()?;
            *prompt = None;
            *prompt = Some(PromptScratch::new(b, &self.persistent, group, slots, self.main_activation_quantization(), mailbox)?);
        }
        let prompt = prompt.as_ref().unwrap();
        // One recurrence span per rank is shared across all prompt slots.
        let bound: usize = self.persistent.iter().map(|w| prompt_bytes(&w.plan, prompt.group, prompt.slots)).sum();
        Ok(bound.saturating_sub(prompt.device_bytes()))
    }
    /// Serving admission funds the largest rank, including uneven KV ownership.
    pub fn request_reservation_bytes(&self, capacity: usize, tile: usize, slots: usize, caching: bool) -> Result<usize> {
        self.request_reservation_bytes_for_depth(capacity, tile, slots, caching, 0)
    }
    fn request_reservation_bytes_for_depth(&self, capacity: usize, tile: usize, slots: usize,
        caching: bool, depth: usize) -> Result<usize> {
        if slots == 0 || slots > GROUPS_IN_FLIGHT { return Err("invalid Flash workspace reservation".into()); }
        let world = self.persistent[0].plan.ranks;
        (0..world).try_fold(0, |maximum, rank| {
            let mut plan = RankPlan::new(rank, world).map_err(|e| e.to_string())?;
            if !plan.is_pair() { plan.expert_output_rows = self.persistent[0].plan.expert_output_rows.clone(); }
            let execution = session_bytes(&plan, capacity, tile, slots, depth + 1)
                + mtp::history_bytes(&plan, capacity, depth);
            // Capacity growth replaces KV and attention buffers, never the
            // whole session. Keep its variable execution bound for old/new
            // overlap; the permanent 256 MiB allowance is funded only once.
            let growth = execution - 256 * 1024 * 1024;
            // A plain cache continuation may alias a former solo owner's MTP
            // snapshots. Keep that immutable allowance until those identities
            // disappear; it does not allocate or maintain live draft history.
            let heads = (plan.value_heads.end - plan.value_heads.start) as usize;
            let keys = (plan.key_heads.end - plan.key_heads.start) as usize;
            let recurrent = 36 * (heads * 128 * 128 + (2 * keys + heads) * 128 * 4) * 4 + Ple::HISTORY_BYTES;
            let snapshot = |tokens| 2 * recurrent + tokens * 12 * (2 * (plan.kv_heads.end - plan.kv_heads.start) as usize * 256 + 128) * 4
                + tokens * 4 + 65536 + if self.mtp_depth == 0 { 0 } else { 2 * mtp::snapshot_bytes(&plan, tokens) };
            // Growth/restore finishes before publication. Its allowance also
            // covers one new full snapshot while old fork aliases are reclaimed.
            let growth = if caching { growth.max(snapshot(capacity)) } else { growth };
            // Retained copies borrow cache headroom instead of admission bytes.
            Ok(maximum.max(execution + growth + self.graph_reservation_bytes_for_depth(depth)))
        })
    }
    pub fn load(b: &Backend<'a>, root: &Path, checkpoint: Checkpoint) -> Result<Self> {
        if checkpoint.config.mixed.is_some() { return Err("mixed execution awaits MXFP8 projections and CSF kernel integration".into()); }
        super::require_inference(checkpoint.config.format)?;
        let activation_quantization = Moe::activation_quantization()?;
        let mut persistent = Vec::with_capacity(2);
        for rank in 0..2 {
            let start = Instant::now();
            persistent.push(Weights::load_selected(b, root, &checkpoint,
                RankPlan::new(rank, 2).map_err(|e| e.to_string())?, |name|
                    name == "model.language_model.embed_tokens.weight" || name == "lm_head.weight"
                    || name.starts_with(PLE) || name.starts_with(MIXER))?);
            println!("flash_session resident rank={rank} weights_bytes={} load_ms={:.3}",
                persistent.last().unwrap().resident_bytes(), start.elapsed().as_secs_f64()*1000.0);
        }
        Ok(Self { root: root.to_owned(), checkpoint, persistent, resident: false, activation_quantization,
            mtp_depth: 0, profile_layers: crate::collective::layers_enabled(), prompt: RefCell::new(None), draft_mx8: None, _lbf12: Default::default() })
    }
    /// Load a single rank for a distributed session. The budget covers weights
    /// and execution memory; a low budget retains the existing streamed path.
    pub fn load_rank(b: &Backend<'a>, root: &Path, checkpoint: Checkpoint,
        rank: usize, world: usize, capacity: usize, tile: usize) -> Result<Self> {
        Self::load_rank_with_mtp(b, root, checkpoint, rank, world, capacity, tile, 0)
    }
    pub fn load_rank_with_mtp(b: &Backend<'a>, root: &Path, checkpoint: Checkpoint,
        rank: usize, world: usize, capacity: usize, tile: usize, mtp_depth: usize) -> Result<Self> {
        if checkpoint.config.mixed.is_some() { return Err("mixed execution awaits MXFP8 projections and CSF kernel integration".into()); }
        let (plan, weight_bytes, reserve) = Self::rank_load_plan(&checkpoint, rank, world, capacity, tile, mtp_depth)?;
        let budget = b.allocation_budget().saturating_sub(b.allocated_bytes());
        let resident = weight_bytes.checked_add(reserve).is_some_and(|n| n <= budget);
        let activation_quantization = Moe::activation_quantization()?;
        println!("flash_session dense_arithmetic=bf16_hi_lo_tensor_core");
        println!("flash_session plan rank={rank} layer_weights={} budget_bytes={budget} planned_weights_bytes={weight_bytes} execution_reserve_bytes={reserve}",
            if resident { "resident" } else { "streamed" });
        if mtp_depth != 0 && !resident {
            return Err(spark_gpu_probe::admission::refusal(weight_bytes as u128 + reserve as u128, budget,
                &format!("weights/rank {}; execution/session {} x 1 session; CONTEXT={capacity}; OUTPUT not supplied; MTP needs resident weights", spark_gpu_probe::admission::bytes(weight_bytes as u128), spark_gpu_probe::admission::bytes(reserve as u128)),
                "the remaining allocation ceiling after bootstrap CUDA/unified-memory safety reserve and live allocations", spark_gpu_probe::admission::FLASH));
        }
        let start = Instant::now();
        let before = b.stats();
        let weights = if resident { Weights::load(b, root, &checkpoint, plan)? }
            else { Weights::load_selected(b, root, &checkpoint, plan, |name|
                name == "model.language_model.embed_tokens.weight" || name == "lm_head.weight"
                || name.starts_with(PLE) || name.starts_with(MIXER))? };
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let stats = b.stats().since(before);
        let io_ms = stats.weight_read_upload_ns as f64 / 1e6;
        println!("flash_session resident rank={rank} weights_bytes={} payload_bytes={} load_ms={load_ms:.3} allocation_ms={:.3} read_upload_ms={io_ms:.3} payload_gbps={:.3} read_upload_gbps={:.3} file_bytes={} file_gbps={:.3} buffered_bytes={} peak_pinned_staging_bytes={} pinned_staging_bytes={} host_mirror_bytes=0",
            weights.resident_bytes(), weights.payload_bytes, stats.allocation_ns as f64 / 1e6,
            weights.payload_bytes as f64 / (load_ms * 1e6),
            weights.payload_bytes as f64 / (io_ms * 1e6), stats.weight_file_bytes,
            stats.weight_file_bytes as f64 / (io_ms * 1e6), stats.weight_buffered_bytes,
            b.peak_pinned_staging_bytes(), b.pinned_staging_bytes());
        Ok(Self { draft_mx8: if mtp_depth != 0 && resident { Head::fp8_draft_twin(b, &weights)? } else { None }, root: root.to_owned(), _lbf12: if resident { super::blocks::Lbf12Twins::fp8(b, &weights, reserve)? } else { Default::default() },
            checkpoint, persistent: vec![weights], resident, activation_quantization, mtp_depth, profile_layers: crate::collective::layers_enabled(), prompt: RefCell::new(None) })
    }
    /// Serving ranks agree on this decision before any potentially slow load.
    /// The caller passes the remaining bytes of the backend's bounded budget.
    pub fn rank_will_be_resident(checkpoint: &Checkpoint, rank: usize, world: usize,
        capacity: usize, tile: usize, mtp_depth: usize, budget: usize) -> Result<bool> {
        let (_, weights, reserve) = Self::rank_load_plan(checkpoint, rank, world, capacity, tile, mtp_depth)?;
        Ok(weights.checked_add(reserve).is_some_and(|n| n <= budget))
    }
    pub fn rank_residency_refusal(checkpoint: &Checkpoint, rank: usize, world: usize,
        capacity: usize, tile: usize, mtp_depth: usize, budget: usize, output: usize) -> Result<String> {
        let (_, weights, reserve) = Self::rank_load_plan(checkpoint, rank, world, capacity, tile, mtp_depth)?;
        Ok(spark_gpu_probe::admission::refusal(weights as u128 + reserve as u128, budget,
            &format!("weights/rank {}; execution/session {} x 1 session; CONTEXT={capacity}; OUTPUT={output}; ranks must agree on residency", spark_gpu_probe::admission::bytes(weights as u128), spark_gpu_probe::admission::bytes(reserve as u128)),
            "the rank's remaining bootstrap allocation ceiling: Linux MemAvailable (page cache counts as free), capped by CUDA total and SEQPAR_MEMORY_BYTES; less 10% reserve (at least 2 GiB) and live allocations", spark_gpu_probe::admission::FLASH))
    }
    fn rank_load_plan(checkpoint: &Checkpoint, rank: usize, world: usize,
        capacity: usize, tile: usize, mtp_depth: usize) -> Result<(RankPlan, usize, usize)> {
        super::require_inference(checkpoint.config.format)?;
        // Verification rows (draft depth + 1) stay within the widest decode launch.
        if mtp_depth >= DecodeBatch::MAX_ROWS { return Err("Flash MTP depth must be 0..=15".into()); }
        let activation_quantization = Moe::activation_quantization()?;
        if tile == 0 || tile > 64 || capacity < tile || capacity > 262144 {
            return Err("Flash session needs a 1..64 token tile and sufficient cache capacity".into());
        }
        let mut plan = RankPlan::new(rank as u64, world as u64).map_err(|e| e.to_string())?;
        plan.mtp = mtp_depth != 0;
        if activation_quantization && checkpoint.config.format == engine_model::flash::WeightFormat::Fp8 {
            let rows = partition(20, rank as u64, world as u64).map_err(|e| e.to_string())?;
            plan.expert_output_rows = Some(rows.start * 128..rows.end * 128);
        }
        let weight_bytes = Weights::planned_bytes(checkpoint, plan.clone())?;
        // Reserve fixed GDN states/workspaces and bounded host staging; all
        // twelve QSA caches, their selection scratch and the attention score
        // buffer (tile x heads x capacity FP32); and the group-token workspaces.
        // Transport is not connected yet for all callers. Reserve the largest
        // shared prompt workspace plus one private execution here;
        // serving admission uses the actual mailbox and residency after loading.
        let reserve = session_bytes(&plan, capacity, tile, 1, mtp_depth + 1)
            + mtp::history_bytes(&plan, capacity, mtp_depth)
            + mtp::reservation_bytes(&plan, capacity, mtp_depth)
            + prompt_bytes(&plan, group_tokens(capacity, tile), GROUPS_IN_FLIGHT)
            + spark_gpu_probe::flash_dense_reserve_bytes();
        Ok((plan, weight_bytes, reserve))
    }
    pub fn session<'w, E: Exchange>(&'w self, b: &Backend<'a>, ranks: Vec<E>,
        capacity: usize, tile: usize) -> Result<Session<'a, 'w, E>> {
        Session::new(b, self, ranks, capacity, tile, None)
    }
    /// Serving sessions share the transport's single host registration.
    pub fn session_with_mailbox<'w, E: Exchange>(&'w self, b: &Backend<'a>, ranks: Vec<E>,
        capacity: usize, tile: usize, mailbox: Option<Rc<HostRegistration<'a>>>) -> Result<Session<'a, 'w, E>> {
        Session::new(b, self, ranks, capacity, tile, mailbox)
    }
    /// Serving opens plain; only its sole owner may enable optional drafting.
    pub fn session_plain_with_mailbox<'w, E: Exchange>(&'w self, b: &Backend<'a>, ranks: Vec<E>,
        capacity: usize, tile: usize, mailbox: Option<Rc<HostRegistration<'a>>>) -> Result<Session<'a, 'w, E>> {
        Session::new_plain(b, self, ranks, capacity, tile, mailbox)
    }
}

fn session_bytes(plan: &RankPlan, capacity: usize, tile: usize, slots: usize, decode_rows: usize) -> usize {
    let keys = (plan.kv_heads.end - plan.kv_heads.start) as usize;
    let heads = (plan.query_heads.end - plan.query_heads.start) as usize;
    256 * 1024 * 1024 + decode_rows * 4 + capacity * 12 * (2 * keys * 256 + 128) * 4
        + (capacity / 4) * (128 + 2 * tile) * 4 + tile * heads * capacity * 4
        + slots * (group_bytes(decode_rows) + if plan.expert_output_rows.is_some() {
            if plan.is_pair() { 2 * (decode_rows * spark_gpu_probe::driver::fp8_abi::MIDDLE_PACKET_WORDS * 4).div_ceil(256) * 256 }
            else { compact_bytes(plan.ranks as usize, decode_rows) }
        } else { 0 })
}

pub struct Output {
    pub token: u32,
    pub logits: Option<Vec<f32>>,
    /// Includes all weight streaming, GPU execution, staging and collectives.
    pub elapsed_ms: f64,
    pub load_ms: f64,
}

/// One reusable, stateless tile for independent decode rows. Attention, lexical
/// histories and token positions remain in their owning Session. No weight copy.
pub struct DecodeBatch<'a, 'w> {
    graphs: Vec<Option<DecodeGraph<'a>>>,
    graph_widths: [usize; 2],
    qsa_shapes: Vec<[usize; 2]>,
    model: &'w Model<'a>,
    descriptors: Vec<DeviceVector<'a>>,
    hidden: DeviceVector<'a>,
    input: DeviceVector<'a>,
    logits: DeviceVector<'a>,
    candidates: DeviceVector<'a>,
    row_wire: DeviceVector<'a>,
    row_copies: DeviceVector<'a>,
    hc: HcScratch<'a>,
    moe: MoeScratch<'a>,
    gdn: GdnBatch<'a>,
    qsa: QsaBatch<'a>,
    capacity: usize,
    prepared_rows: usize,
    last_collectives: Option<u64>,
    routing: Fp8RoutingFlag<'a>,
}
impl<'a, 'w> DecodeBatch<'a, 'w> {
    pub const MAX_ROWS: usize = super::blocks::DECODE_ROWS;
    /// Keep the two most recently replayed cohort widths. Stable batches retain
    /// their existing graph; switching to a third evicts its oldest variant.
    pub fn graph_reservation_bytes(model: &Model<'_>) -> usize {
        if !model.resident { return 0; }
        let segments = if model.persistent[0].plan.expert_output_rows.is_some() { 4 } else { 3 };
        2 * model.checkpoint.config.layers.len() * segments * DecodeGraph::reservation_bytes_for_rows(Self::MAX_ROWS)
    }
    pub fn reservation_bytes(model: &Model<'_>) -> usize {
        let descriptor = if model.checkpoint.config.format == engine_model::flash::WeightFormat::Nvfp4 {
            std::mem::size_of::<spark_gpu_probe::driver::fp4_abi::Expert>()
        } else { 7 * 8 };
        Self::MAX_ROWS * (GROUP_TOKEN_BYTES + VOCABULARY * 4 + 4) + EXPERT_PADDING_BYTES + 48 * 512 * descriptor
            + if model.persistent[0].plan.expert_output_rows.is_some() { compact_bytes(model.persistent[0].plan.ranks as usize, Self::MAX_ROWS) } else { 0 }
            // Upper bound at world=1: GDN QKV/Z/A/B/gated and QSA QG/K/V/index/gated.
            + Self::MAX_ROWS * (10240 + 6144 + 96 + 6144 + 12288 + 512 + 512 + 640 + 6144) * 4 + 640
            + Self::MAX_ROWS * std::mem::size_of::<spark_gpu_probe::gdn_params::GdnSessionRow>()
            + Self::MAX_ROWS * std::mem::size_of::<spark_gpu_probe::kernel_params::QsaBatchRow>()
            + Self::MAX_ROWS * (24 + EMBEDDING_BYTES + LEXICAL_BYTES + 17 * std::mem::size_of::<spark_gpu_probe::kernel_params::HybridCopyRow>())
    }
    /// Owned device buffers, including reserved row capacity and resident expert
    /// descriptors. The model is borrowed; graphs reference these same buffers.
    /// Driver graph storage is funded separately by graph_reservation_bytes and
    /// included in the backend's accounted-byte ceiling.
    pub fn device_bytes(&self) -> usize {
        self.routing.bytes()
            + self.capacity * (H + W) * 4 + (self.logits.len() + self.candidates.len()) * 4
            + self.capacity * (EMBEDDING_BYTES + LEXICAL_BYTES + 17 * std::mem::size_of::<spark_gpu_probe::kernel_params::HybridCopyRow>())
            + self.hc.device_bytes() + self.moe.device_bytes()
            + self.descriptors.iter().map(|v| v.len() * 4).sum::<usize>() + self.gdn.device_bytes() + self.qsa.device_bytes()
    }
    pub fn new(b: &Backend<'a>, model: &'w Model<'a>) -> Result<Self> {
        let capacity = Self::MAX_ROWS;
        let layers = model.checkpoint.config.layers.len();
        let descriptors = if model.resident {
            (0..layers).map(|layer| Moe::load(&model.persistent[0], layer, model.activation_quantization)?
                .device_descriptors(b)).collect::<Result<Vec<_>>>()?
        } else { Vec::new() };
        let segments = if model.persistent[0].plan.expert_output_rows.is_some() { 4 } else { 3 };
        Ok(Self { graphs: (0..capacity * layers * segments * if b.ab_enabled() { 2 } else { 1 }).map(|_| None).collect(), graph_widths: [0; 2], model, descriptors,
            qsa_shapes: vec![[0; 2]; capacity * layers],
            hidden: b.vector(capacity * W)?, input: b.vector(capacity * H)?,
            logits: b.vector(capacity * ((model.persistent[0].plan.vocabulary.end - model.persistent[0].plan.vocabulary.start) as usize + 3))?,
            candidates: b.vector(capacity * 3)?,
            row_wire: b.vector(capacity * (EMBEDDING_BYTES + LEXICAL_BYTES) / 4)?,
            row_copies: b.vector(capacity * 17 * std::mem::size_of::<spark_gpu_probe::kernel_params::HybridCopyRow>() / 4)?,
            hc: HcScratch::with_tokens(b, capacity)?, moe: MoeScratch::with_capacity(b, capacity)?,
            gdn: GdnBatch::new(b, &model.persistent[0], capacity)?,
            qsa: QsaBatch::new(b, &model.persistent[0].plan, capacity)?,
            capacity, prepared_rows: 0, last_collectives: None, routing: Fp8RoutingFlag::new(b, capacity)? })
    }
    /// Transport's existing counter delta for the last fixed-row traversal,
    /// including a completed output gather that reported rank disagreement.
    /// None means the transport does not expose counters or no batch ran.
    /// This reads no GPU data and introduces no transport message.
    pub fn collective_count(&self) -> Option<u64> { self.last_collectives }
    fn retain_graph_width(&mut self, b: &Backend<'a>, width: usize) -> Result<()> {
        if self.graph_widths[0] == width { return Ok(()); }
        if self.graph_widths[1] != width && self.graph_widths[1] != 0 {
            b.synchronize()?;
            let layers = self.model.checkpoint.config.layers.len();
            let segments = if self.model.persistent[0].plan.expert_output_rows.is_some() { 4 } else { 3 };
            let base = (self.graph_widths[1] - 1) * layers * segments;
            for graph in &mut self.graphs[base..base + layers * segments] { *graph = None; }
            if b.ab_enabled() {
                let base = base + self.graphs.len() / 2;
                for graph in &mut self.graphs[base..base + layers * segments] { *graph = None; }
            }
        }
        self.graph_widths = [width, self.graph_widths[0]];
        Ok(())
    }
    fn segment(&mut self, b: &Backend<'a>, key: Option<usize>, run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        if b.is_profiling() { return run(self); }
        let Some(key) = key else { return run(self) };
        let arm_key = key + b.arm_index() * (self.graphs.len() / 2);
        if self.graphs[arm_key].is_none() {
            // Session addresses are read from the stable batch descriptor table
            // at replay time. Captured addresses belong to the batch or model.
            let segments = if self.model.persistent[0].plan.expert_output_rows.is_some() { 4 } else { 3 };
            let width = key / (self.model.checkpoint.config.layers.len() * segments) + 1;
            self.graphs[arm_key] = Some(unsafe { b.capture_batch_bounded(DecodeGraph::node_limit(width), || run(self))? });
        }
        unsafe { b.replay_decode(self.graphs[arm_key].as_ref().unwrap()) }
    }
}

pub struct Session<'a, 'w, E> {
    /// The page-locked mailbox of a distributed rank; dropped before `ranks`.
    mailbox: Option<Rc<HostRegistration<'a>>>,
    model: &'w Model<'a>,
    ranks: Vec<Profiled<E>>,
    rows: Vec<Rows>,
    front: Vec<FrontScratch<'a>>,
    ple: Vec<Ple<'a, 'w>>,
    /// Group workspaces (hc, moe, input, hidden, exchange): one per group in
    /// flight and local rank, index group * local + rank. Decode uses group 0.
    hc: Vec<HcScratch<'a>>,
    head: Vec<Head<'a, 'w>>,
    moe: Vec<MoeScratch<'a>>,
    resident_moe: Vec<Vec<Moe<'w>>>,
    /// Device expert descriptors per layer (28,672 FP8 / 40,960 NVFP4 bytes).
    moe_descriptors: Vec<Vec<DeviceVector<'a>>>,
    gdn: Vec<Option<GdnScratch<'a>>>,
    /// Swapped in from the model only while executing a prompt.
    gdn_group: Vec<GdnScratch<'a>>,
    gdn_verify: Vec<Option<GdnScratch<'a>>>,
    qsa: Vec<QsaScratch<'a>>,
    gdn_cache: Vec<Vec<Option<GdnState<'a>>>>,
    qsa_cache: Vec<Vec<Option<QsaState<'a>>>>,
    /// Constant per-layer norm vectors, assembled across ranks on first use
    /// (GDN norms kept on the device and copied into the workspace per layer).
    gdn_norms: Vec<Vec<Option<DeviceVector<'a>>>>,
    qsa_norms: Vec<Vec<Option<DeviceVector<'a>>>>,
    qsa_shapes: Vec<[usize; 3]>,
    input: Vec<DeviceVector<'a>>,
    hidden: Vec<DeviceVector<'a>>,
    /// Pinned exchange slot per group workspace, sized for its MoE partials.
    exchange: Vec<ExchangeSlot<'a>>,
    /// Four groups for resident host reductions; one for duplex or streaming.
    slots: usize,
    communication: Vec<Vec<f32>>,
    norm_offsets: Vec<usize>,
    candidate_offsets: Vec<usize>,
    routing: Vec<Fp8RoutingFlag<'a>>,
    vocabulary_offsets: Vec<usize>,
    capacity: usize,
    tile: usize,
    group: usize,
    position: usize,
    prefill_group: Option<usize>,
    valid: bool,
    weight_peak: [usize; 2],
    cache_bytes: [usize; 2],
    cache_peak: [usize; 2],
    scratch_peak: [usize; 2],
    shared_decode_peak: usize,
    device_peak: [usize; 2],
    total_peak: usize,
    /// Per-layer decode/verification segments between collectives of a resident
    /// distributed rank, captured on their first use after a prompt.
    graphs: Vec<Option<DecodeGraph<'a>>>,
    caching: bool,
    canonical: Option<Rc<Recurrent<'a>>>,
    prefill_tail: Vec<u32>,
    mtp: Vec<Draft<'a, 'w>>,
    mtp_enabled: bool,
    mtp_workspace: bool,
    mtp_failed: bool,
    verification: Option<Verification<'a>>,
    verifying: bool,
    rollback: Vec<Recurrent<'a>>,
    verify_logits: Vec<DeviceVector<'a>>,
    verify_staging: Vec<Vec<DeviceVector<'a>>>,
}
impl<'a, 'w, E: Exchange> Session<'a, 'w, E> {
    pub fn capacity(&self) -> usize { self.capacity }
    pub fn is_valid(&self) -> bool { self.valid }
    /// Limit prompt row groups without changing the 64-token attention tile or
    /// arithmetic. The existing shared workspace remains the allocation bound.
    pub fn set_prefill_group(&mut self, tokens: usize) -> Result<()> {
        let maximum = group_tokens(self.capacity, self.tile);
        if self.tile != 64 || tokens == 0 || tokens % 64 != 0 || tokens > maximum {
            return Err(format!("Flash prefill group must be a multiple of 64 within 64..={maximum}"));
        }
        self.prefill_group = Some(tokens);
        Ok(())
    }
    /// Request-funded growth preserves every committed cache row. Drop graphs
    /// before replacing buffers; prompt grouping remains fixed after 512 tokens.
    pub fn reserve(&mut self, b: &Backend<'a>, capacity: usize) -> Result<()> {
        if capacity <= self.capacity { return Ok(()); }
        if capacity > 262144 || self.capacity < 512 { return Err("invalid Flash serving cache growth".into()); }
        self.commit_verification(b, self.position)?;
        self.release_mtp_workspace(b)?;
        self.drop_graphs();
        for r in 0..self.ranks.len() {
            for state in self.qsa_cache[r].iter_mut().flatten() { *state = state.snapshot(b, capacity)?; }
            self.qsa[r] = QsaScratch::new(b, &self.model.persistent[r].plan, self.tile, capacity)?;
            if let Some(draft) = self.mtp.get_mut(r) { draft.reserve(b, &self.model.persistent[r].plan, capacity)?; }
            self.cache_bytes[r] = Ple::HISTORY_BYTES
                + self.gdn_cache[r].iter().flatten().map(GdnState::device_bytes).sum::<usize>()
                + self.qsa_cache[r].iter().flatten().map(QsaState::device_bytes).sum::<usize>();
        }
        self.capacity = capacity;
        Ok(())
    }
    fn new(b: &Backend<'a>, model: &'w Model<'a>, ranks: Vec<E>, capacity: usize, tile: usize,
        mailbox: Option<Rc<HostRegistration<'a>>>) -> Result<Self> {
        Self::new_with_drafting(b, model, ranks, capacity, tile, mailbox, true)
    }
    fn new_plain(b: &Backend<'a>, model: &'w Model<'a>, ranks: Vec<E>, capacity: usize, tile: usize,
        mailbox: Option<Rc<HostRegistration<'a>>>) -> Result<Self> {
        Self::new_with_drafting(b, model, ranks, capacity, tile, mailbox, false)
    }
    fn new_with_drafting(b: &Backend<'a>, model: &'w Model<'a>, ranks: Vec<E>, capacity: usize, tile: usize,
        mailbox: Option<Rc<HostRegistration<'a>>>, drafting: bool) -> Result<Self> {
        if ranks.len() != model.persistent.len() || tile == 0 || tile > 64 || capacity < tile || capacity > 262144 {
            return Err("Flash session needs matching local ranks, a 1..64 token tile and sufficient cache capacity".into());
        }
        let local = ranks.len();
        let mtp_enabled = drafting && model.mtp_depth != 0;
        let group = if mtp_enabled { model.mtp_depth + 1 } else { 1 };
        let world = model.persistent[0].plan.ranks as usize;
        let offsets = |width| -> Result<Vec<usize>> {
            let mut offsets = vec![0];
            for rank in 0..world {
                offsets.push(partition(width, rank as u64, world as u64)
                    .map_err(|e| e.to_string())?.end as usize);
            }
            Ok(offsets)
        };
        let mut ranks = ranks.into_iter().map(Profiled::new).collect::<Vec<_>>();
        // Safety: an enabled mailbox never moves and outlives this session's registration.
        let mailbox = if mailbox.is_some() { mailbox } else { match ranks.as_mut_slice() {
            [rank] => rank.mailbox_memory().map(|(base, bytes)| unsafe { b.register_host(base, bytes) }.map(Rc::new)).transpose()?,
            _ => None,
        } };
        // The duplex GPU slot path stays synchronous; host reductions overlap groups.
        let slots = model.workspace_slots(mailbox.is_some());
        model.prepare_prompt(b, capacity, tile, mailbox.is_some())?;
        let mut result = Self { mailbox, model, ranks, rows: Vec::new(), front: Vec::new(), ple: Vec::new(),
            hc: Vec::new(), head: Vec::new(), moe: Vec::new(), resident_moe: Vec::new(), moe_descriptors: Vec::new(), gdn: (0..local).map(|_| None).collect(), gdn_group: Vec::new(), qsa: Vec::new(),
            gdn_cache: (0..local).map(|_| (0..48).map(|_| None).collect()).collect(),
            qsa_cache: (0..local).map(|_| (0..48).map(|_| None).collect()).collect(),
            gdn_norms: (0..local).map(|_| (0..48).map(|_| None).collect()).collect(), qsa_norms: (0..local).map(|_| (0..48).map(|_| None).collect()).collect(), qsa_shapes: vec![[0; 3]; group],
            graphs: (0..forward::decode_graph_slots(model, group)).map(|_| None).collect(),
            caching: false, canonical: None, prefill_tail: Vec::new(),
            mtp: Vec::new(), mtp_enabled, mtp_workspace: false, mtp_failed: false,
            verification: None, verifying: false, rollback: Vec::new(), verify_logits: Vec::new(), verify_staging: Vec::new(),
            gdn_verify: (0..local).map(|_| None).collect(),
            input: Vec::new(), hidden: Vec::new(), exchange: Vec::new(),
            slots, communication: vec![vec![0.0; world*tile*11*H]; local],
            routing: (0..local).map(|_| Fp8RoutingFlag::new(b, group)).collect::<Result<Vec<_>>>()?,
            norm_offsets: offsets(128)?, candidate_offsets: (0..=world).map(|r| r * 2).collect(),
            vocabulary_offsets: offsets(VOCABULARY as u64)?,
            capacity, tile, group,
            position: 0, prefill_group: None, valid: true, weight_peak: [0;2],
            cache_bytes: [Ple::HISTORY_BYTES;2], cache_peak: [0;2], scratch_peak: [0;2], shared_decode_peak: 0, device_peak: [0;2], total_peak: 0 };
        for w in &model.persistent {
            if mtp_enabled { result.mtp.push(Draft::new(b, w, capacity, model.activation_quantization)?); }
            result.rows.push(Rows::new(w)?);
            result.front.push(FrontScratch::with_tokens(b, result.group)?);
            result.ple.push(Ple::load(b, w)?);
            result.head.push(Head::load(b, w)?);
            result.head.last_mut().unwrap().use_mx8_draft(model.draft_mx8.as_ref().filter(|_| local == 1));
            result.resident_moe.push(if model.resident {
                (0..model.checkpoint.config.layers.len()).map(|layer| Moe::load(w, layer, model.activation_quantization))
                    .collect::<Result<Vec<_>>>()?
            } else { Vec::new() });
            result.moe_descriptors.push(result.resident_moe.last().unwrap().iter()
                .map(|layer| layer.device_descriptors(b)).collect::<Result<Vec<_>>>()?);
            result.qsa.push(QsaScratch::new(b, &w.plan, tile, capacity)?);
        }
        for _ in 0..result.slots * local {
            result.hc.push(HcScratch::with_tokens(b, result.group)?);
            result.moe.push(MoeScratch::with_capacity(b, result.group)?);
            result.moe.last_mut().unwrap().reserve_grouped(b, result.group, model.main_activation_quantization(),
                model.persistent[0].plan.expert_output_rows.as_ref().map(|_| model.persistent[0].plan.ranks as usize))?;
            let (input, hidden) = execution_rows(b, result.group)?;
            result.input.push(input);
            result.hidden.push(hidden);
            result.exchange.push(b.exchange_slot(result.group * PARTIALS)?);
        }
        result.account(b, [0;2], 0);
        Ok(result)
    }
    /// Reset state and both EOS/ngram and PLE-convolution histories together.
    /// Dropping detached caches also clears them after a partially failed call.
    pub fn reset(&mut self, b: &Backend<'a>) -> Result<()> {
        self.drop_graphs();
        self.canonical = None;
        self.prefill_tail.clear();
        self.verification = None;
        self.mtp_workspace = false;
        self.mtp_failed = false;
        self.rollback.clear();
        self.verify_logits.clear();
        self.verify_staging.clear();
        for scratch in &mut self.gdn_verify { *scratch = None; }
        self.mtp.clear();
        for w in &self.model.persistent {
            if self.mtp_enabled { self.mtp.push(Draft::new(b, w, self.capacity, self.model.activation_quantization)?); }
        }
        for r in 0..self.ranks.len() {
            self.rows[r].reset();
            self.ple[r].reset(b)?;
            for state in &mut self.gdn_cache[r] { *state = None; }
            for state in &mut self.qsa_cache[r] { *state = None; }
        }
        for moe in &mut self.moe { *moe = MoeScratch::with_capacity(b, self.group)?; }
        self.cache_bytes = [Ple::HISTORY_BYTES;2];
        self.position = 0;
        self.valid = true;
        Ok(())
    }
    /// Captured graphs hold workspace addresses: drop them before any prompt
    /// (workspaces may grow) and on reset (MoE workspaces are replaced).
    fn drop_graphs(&mut self) {
        for graph in &mut self.graphs { *graph = None; }
        for draft in &mut self.mtp { draft.drop_graphs(); }
    }
    /// Run a decode segment between collectives: eagerly without a key, else
    /// replay its graph, capturing it first (capture records the same
    /// launches with the same arguments; it does not execute them).
    fn segment(&mut self, b: &Backend<'a>, key: Option<usize>, run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        if b.is_profiling() { return run(self); }
        let Some(key) = key else { return run(self) };
        if self.graphs[key].is_none() {
            // Safety: captured kernels read session workspaces, caches and
            // resident weights, which stay allocated until drop_graphs.
            let segments = if self.model.persistent[0].plan.expert_output_rows.is_some() { 3 } else { 2 };
            let width = key / (48 * segments + 24) + 1;
            let graph = unsafe { b.capture_batch_bounded(DecodeGraph::node_limit(width), || run(self))? };
            self.graphs[key] = Some(graph);
        }
        unsafe { b.replay_decode(self.graphs[key].as_ref().unwrap()) }
    }
    /// Scratch is execution state, never part of a checkpoint. A restored
    /// session needs the same allocation as a cold session before attention.
    /// This uses only rank dimensions, including for streamed weights, and
    /// performs no collective so restore remains a local operation.
    fn prepare_attention_scratch(&mut self, b: &Backend<'a>) -> Result<()> {
        for (scratch, weights) in self.gdn.iter_mut().zip(&self.model.persistent) {
            if scratch.is_none() {
                *scratch = Some(GdnScratch::new(b, weights, self.tile, &[0.0; 128])?);
            }
        }
        Ok(())
    }
    /// Both solo/prompt and batched attention enter through this preparation.
    /// Restored histories do not imply that this session gathered the layer's
    /// norms. Finish lazy allocation and collectives before graph capture.
    fn prepare_attention(&mut self, b: &Backend<'a>, weights: &[Weights<'_>], layer: usize) -> Result<()> {
        if self.model.checkpoint.config.layers[layer] == LayerKind::DeltaNet {
            self.prepare_attention_scratch(b)?;
            self.gdn_norm(b, weights, layer)?;
        } else if !self.qsa_norms.iter().all(|norms| norms[layer].is_some()) {
            let gathered = weights.iter().map(|w| QsaGather::norms(w, layer)).collect::<Result<Vec<_>>>()?;
            b.prefill_read_bytes(9, || gathered.iter().map(QsaGather::host_bytes).sum());
            let norms = self.gather_qsa(gathered)?;
            for (cached, norm) in self.qsa_norms.iter_mut().zip(norms) {
                let mut device = b.vector_arena(&[768])?.remove(0);
                b.write_vector(&mut device, &norm)?;
                cached[layer] = Some(device);
            }
        }
        Ok(())
    }
    /// Assemble a GDN layer's norm across ranks once and keep a device copy.
    fn gdn_norm(&mut self, b: &Backend<'a>, weights: &[Weights<'_>], layer: usize) -> Result<()> {
        if self.gdn_norms.iter().all(|norms| norms[layer].is_some()) { return Ok(()); }
        let local = self.ranks.len();
        let name = format!("model.language_model.layers.{layer}.linear_attn.norm.weight");
        let mut norms = vec![[0.0f32; 128]; local];
        for r in 0..local {
            let tensor = weights[r].tensor(&name)?;
            for (i, value) in weights[r].values_f32(&name)?.into_iter().enumerate() {
                norms[r][tensor.global_byte_offset((i*2) as u64)? as usize/2] = value;
            }
        }
        let offsets = &self.norm_offsets;
        b.prefill_read_bytes(9, || norms.iter().map(std::mem::size_of_val).sum());
        let norms = parallel(&mut self.ranks, norms, |rank, mut norm| {
            rank.gather_f32(&mut norm, offsets)?; Ok(norm)
        })?;
        for r in 0..local {
            let mut device = b.vector(128)?;
            b.write_vector(&mut device, &norms[r])?;
            self.gdn_norms[r][layer] = Some(device);
        }
        Ok(())
    }
    fn account(&mut self, b: &Backend<'_>, active: [usize;2], shared_decode: usize) {
        self.shared_decode_peak = self.shared_decode_peak.max(shared_decode);
        let local = self.ranks.len();
        for r in 0..local {
            self.weight_peak[r] = self.weight_peak[r].max(self.model.persistent[r].resident_bytes() + active[r]);
            let mtp_cache = self.mtp.get(r).map_or(0, Draft::cache_bytes);
            let rollback = self.rollback.iter().chain(self.verification.iter().flat_map(|v| &v.saved))
                .map(|p| p.ple.get(r).map_or(0, |v| v.len() * 4)
                    + p.gdn[r].iter().flatten().map(GdnState::device_bytes).sum::<usize>()).sum::<usize>();
            let cache = self.cache_bytes[r] + mtp_cache + rollback;
            self.cache_peak[r] = self.cache_peak[r].max(cache);
            let scratch = self.mtp.get(r).map_or(0, Draft::scratch_bytes)
                + self.gdn_verify[r].as_ref().map_or(0, GdnScratch::device_bytes)
                + self.verify_logits.get(r).map_or(0, |v| v.len() * 4)
                + self.verify_staging.get(r).map_or(0, |v| v.iter().map(|v| v.len() * 4).sum::<usize>())
                + self.front[r].device_bytes() + Ple::DEVICE_BYTES - Ple::HISTORY_BYTES
                + self.routing[r].bytes() + self.head[r].device_bytes() + self.qsa[r].device_bytes()
                + self.gdn[r].as_ref().map_or(0, |s| s.device_bytes())
                + self.moe_descriptors[r].iter().map(|v| v.len() * 4).sum::<usize>()
                + self.gdn_norms[r].iter().flatten().map(|v| v.len() * 4).sum::<usize>()
                + self.qsa_norms[r].iter().flatten().map(|v| v.len() * 4).sum::<usize>()
                + (0..self.slots).map(|g| self.hc[g * local + r].device_bytes() + self.moe[g * local + r].device_bytes()
                    + self.group * (H + W) * 4).sum::<usize>()
                + self.gdn_group.get(r).map_or(0, GdnScratch::device_bytes) + self.shared_decode_peak
                + if r == 0 { b.flash_dense_bytes() } else { 0 };
            self.scratch_peak[r] = self.scratch_peak[r].max(scratch);
            self.device_peak[r] = self.device_peak[r].max(self.model.persistent[r].resident_bytes()
                + active[r] + cache + scratch);
        }
        self.total_peak = self.total_peak.max(b.allocated_bytes());
    }
    pub fn report(&self) {
        // Session peaks include shared prompt and batch workspaces. Count them
        // (and resident weights) only once when combining concurrent reports.
        for r in 0..self.ranks.len() {
            println!("flash_session memory rank={} peak_weights_bytes={} peak_cache_bytes={} peak_scratch_bytes={} peak_device_bytes={} peak_shared_decode_bytes={} graph_driver_bytes=unmeasured host_communication_bytes={} pinned_exchange_bytes={} host_weight_mirror_bytes=0 loader_host_staging_cap_bytes=8388608",
                self.model.persistent[r].plan.rank, self.weight_peak[r], self.cache_peak[r], self.scratch_peak[r],
                self.device_peak[r], self.shared_decode_peak, self.communication[r].len()*4,
                (0..self.slots).map(|g| self.exchange[g * self.ranks.len() + r].bytes()).sum::<usize>());
        }
        println!("flash_session local_peak_device_bytes={} execution={} layer_weights={} precision={} other_activations=fp32 activation_quantization={}", self.total_peak,
            if self.ranks.len() == 2 { "sequential_two_local_ranks_one_spark" } else { "distributed_rank" },
            if self.model.resident { "resident" } else { "streamed" },
            self.model.expert_precision(), self.model.main_activation_quantization());
    }
    fn gather_qsa(&mut self, values: Vec<QsaGather>) -> Result<Vec<Vec<f32>>> {
        if self.ranks.len() == 1 {
            let rank = &mut self.ranks[0];
            return values.into_iter().map(|value| value.gather(|v, o| rank.gather_f32(v, o))).collect();
        }
        std::thread::scope(|s| {
            let jobs = self.ranks.iter_mut().zip(values).map(|(rank, value)|
                s.spawn(move || value.gather(|v, o| rank.gather_f32(v, o)))).collect::<Vec<_>>();
            jobs.into_iter().map(|j| j.join().map_err(|_| "Flash gather worker panicked")?).collect()
        })
    }
    /// Run an initial prompt, prompt continuation, or cached decode token.
    pub fn forward(&mut self, b: &Backend<'a>, tokens: &[u32], save_logits: bool) -> Result<Output> {
        self.commit_verification(b, self.position)?;
        // About 110 collectives per token each wait for a short GPU segment;
        // spinning is scoped to this call.
        b.set_spin_waits(true);
        let prefill = self.position == 0 || tokens.len() > 1;
        if !prefill { b.decode_profile_begin(0, self.model.persistent[0].plan.rank as usize)?; }
        let output = if prefill {
            self.forward_prefill(b, tokens, save_logits)
        } else { self.forward_tile(b, &[tokens], save_logits, false, None, None, None) };
        b.set_spin_waits(false);
        if !prefill { b.decode_profile_end(output.is_ok())?; }
        if self.caching && !prefill && output.is_ok() { self.prefill_tail.extend_from_slice(tokens); }
        output
    }
    fn collect_output(&mut self, b: &Backend<'a>, save_logits: bool) -> Result<(u32, Option<Vec<f32>>)> {
        let mut candidates = Vec::new();
        let mut logits = Vec::new();
        for r in 0..self.ranks.len() {
            let candidate = self.head[r].local_max(b);
            let candidate = if self.routing[r].failed_after_readback()? {
                Candidate { token: u32::MAX, logit: 0.0 }
            } else { match candidate {
                Err(error) if error.starts_with("nonfinite") => Candidate { token: NONFINITE_LOGIT, logit: 0.0 },
                result => result?.ok_or("empty rank vocabulary")?,
            } };
            let mut wire = vec![0.0; *self.candidate_offsets.last().unwrap()];
            let global_rank = self.model.persistent[r].plan.rank as usize;
            wire[global_rank*2] = f32::from_bits(candidate.token);
            wire[global_rank*2+1] = candidate.logit;
            candidates.push(wire);
        }
        b.prefill_read_bytes(9, || candidates.iter().map(|v| v.len() * 4).sum());
        let candidates = parallel(&mut self.ranks,candidates,|rank,mut wire| {
            rank.gather_f32(&mut wire,&self.candidate_offsets)?; Ok(wire)
        })?;
        // Every rank reduces the same gathered candidates with the same pure
        // function, so the winner needs no further broadcast.
        let winners = candidates.into_iter().map(|wire| Candidate::reduce(wire.chunks_exact(2)
            .map(|p| Candidate {token:p[0].to_bits(),logit:p[1]})).map(|c|c.token)).collect::<Result<Vec<_>>>()?;
        if winners.windows(2).any(|w| w[0] != w[1]) { return Err("local ranks reduced different Flash winners".into()); }
        for r in 0..self.ranks.len() {
            if save_logits {
                let mut wire = vec![0.0;VOCABULARY];
                let range = self.head[r].vocabulary();
                wire[range.start as usize..range.end as usize].copy_from_slice(&self.head[r].logits(b)?);
                logits.push(wire);
            }
        }
        let logits = if save_logits {
            let mut rows = parallel(&mut self.ranks,logits,|rank,mut row| {
                rank.gather_f32(&mut row,&self.vocabulary_offsets)?; Ok(row)
            })?;
            Some(rows.remove(0))
        } else { None };
        b.synchronize()?;
        Ok((winners[0], logits))
    }
    /// All row reductions finish before one readback and one rank gather.
    /// Candidate comparison and vocabulary coordinates are the solo path's.
    /// Greedy rows transfer only their device winner; full vocabulary shards
    /// cross the host and the ring only for the rows that sample.
    fn collect_batch_output(b: &Backend<'a>, batch: &mut DecodeBatch<'a, 'w>, sessions: &mut [&mut Self],
        save_logits: &[bool], active: u16) -> Result<Vec<Result<(u32, Option<Vec<f32>>)>>> {
        let count = sessions.len();
        let lead = &sessions[0];
        let range = lead.head[0].vocabulary();
        let rows = range.len();
        let vocabulary = lead.vocabulary_offsets.clone();
        let rank = lead.model.persistent[0].plan.rank as usize;
        let sampled = (0..count).filter(|&i| save_logits[i]).collect::<Vec<_>>();
        for (i, s) in sessions.iter_mut().enumerate() { s.head[0].enqueue_max(b, &mut batch.candidates, i)?; }
        // Keep each row's argmax validity word through the existing readback.
        let values = if sampled.is_empty() {
            b.read_argmax_raw(&batch.candidates, count, |values| Ok(values.to_vec()))?
        } else {
            for (j, &i) in sampled.iter().enumerate() {
                if i != j { sessions[i].head[0].copy_logits(b, &mut batch.logits, j * rows)?; }
            }
            let shards = sampled.len() * rows;
            b.copy_range(&batch.candidates, &mut batch.logits, 0, shards, count * 3)?;
            b.read_vector_range(&batch.logits, 0, shards + count * 3)?
        };
        let (logits, candidates) = values.split_at(sampled.len() * rows);
        let mut offsets = vec![0];
        for v in vocabulary.windows(2) {
            offsets.push(offsets.last().unwrap() + 1 + count * 2 + sampled.len() * (v[1] - v[0]));
        }
        let mut wire = vec![0.0; *offsets.last().unwrap()];
        let mut routing = RoutingVerdict { active, rejected: 0 };
        for (i, candidate) in candidates.chunks_exact(3).enumerate() {
            if batch.routing.row_failed_after_readback(i)? { routing.rejected |= 1 << i; }
            let invalid = candidate.iter().any(|v| !v.is_finite()) || candidate[2] != 1.0;
            // Routing has its own bitmap. An invalid vocabulary shard remains
            // a row output error and cannot impersonate a routing vote.
            wire[offsets[rank] + 1 + i * 2] = f32::from_bits(if invalid { NONFINITE_LOGIT }
                else { range.start + candidate[1] as u32 });
            wire[offsets[rank] + 1 + i * 2 + 1] = candidate[0];
        }
        wire[offsets[rank]] = f32::from_bits(routing.word());
        if !sampled.is_empty() { wire[offsets[rank] + 1 + count * 2..offsets[rank + 1]].copy_from_slice(logits); }
        sessions[0].ranks[0].gather_f32(&mut wire, &offsets)?;
        let survivors = RoutingVerdict::agree(offsets[..offsets.len() - 1].iter().map(|&at| wire[at].to_bits()), count)?;
        Ok((0..count).map(|i| {
            if survivors & (1 << i) == 0 { return Err(ROUTING_ERROR.into()); }
            let winner = Candidate::reduce(offsets[..offsets.len() - 1].iter().map(|&at|
                Candidate { token: wire[at + 1 + i * 2].to_bits(), logit: wire[at + 1 + i * 2 + 1] }))?;
            let logits = sampled.iter().position(|&s| s == i).map(|j| {
                let mut row = vec![0.0; VOCABULARY];
                for (r, v) in vocabulary.windows(2).enumerate() {
                    let start = offsets[r] + 1 + count * 2 + j * (v[1] - v[0]);
                    row[v[0]..v[1]].copy_from_slice(&wire[start..start + v[1] - v[0]]);
                }
                row
            });
            Ok((winner.token, logits))
        }).collect())
    }
    fn collect_top_k(&mut self, b: &Backend<'a>, k: usize) -> Result<Vec<(u32, f32)>> {
        if !(1..=64).contains(&k) { return Err("invalid Flash top-k width".into()); }
        let world = self.model.persistent[0].plan.ranks as usize;
        let offsets = (0..=world).map(|r| r * k * 2).collect::<Vec<_>>();
        let mut wires = Vec::new();
        for (r, head) in self.head.iter_mut().enumerate() {
            let mut wire = vec![f32::NEG_INFINITY; world * k * 2];
            let rank = self.model.persistent[r].plan.rank as usize;
            let values = head.top_k(b, k);
            let values = if self.routing[r].failed_after_readback()? {
                vec![(u32::MAX, 0.0)]
            } else { match values {
                Err(error) if error.starts_with("nonfinite") => vec![(NONFINITE_LOGIT, 0.0)],
                result => result?,
            } };
            for (dst, (id, value)) in wire[rank*k*2..(rank+1)*k*2].chunks_exact_mut(2).zip(values) {
                dst.copy_from_slice(&[f32::from_bits(id), value]);
            }
            wires.push(wire);
        }
        let wires = parallel(&mut self.ranks, wires, |rank, mut wire| {
            rank.gather_f32(&mut wire, &offsets)?; Ok(wire)
        })?;
        // The first candidate in each shard carries either its best token or
        // an error sentinel. Share solo/MTP classification before sampling.
        Candidate::reduce(wires[0].chunks_exact(k * 2).map(|v|
            Candidate { token: v[0].to_bits(), logit: v[1] }))?;
        if wires[0].chunks_exact(2).any(|v| v[1].is_nan()) {
            return Err("invalid Flash top-k candidate".into());
        }
        let mut values = wires[0].chunks_exact(2).filter(|v| v[1] != f32::NEG_INFINITY)
            .map(|v| (v[0].to_bits(), v[1])).collect::<Vec<_>>();
        values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        values.truncate(k);
        Ok(values)
    }
    fn collect_sample_row(&mut self, b: &Backend<'a>, k: usize) -> Result<SampleRow> {
        if k != 0 { self.collect_top_k(b, k).map(SampleRow::TopK) }
        else { self.collect_output(b, true)?.1.map(SampleRow::Logits).ok_or_else(|| "missing Flash sample logits".into()) }
    }
    fn draft_choice(&mut self, choice: Option<u32>) -> Result<Option<u32>> {
        let mut wires = Vec::new();
        for weights in &self.model.persistent {
            let mut wire = vec![0.0; *self.candidate_offsets.last().unwrap()];
            if weights.plan.rank == 0 {
                if let Some(token) = choice { wire[..2].copy_from_slice(&[f32::from_bits(token), 1.0]); }
            }
            wires.push(wire);
        }
        let wires = parallel(&mut self.ranks, wires, |rank, mut wire| {
            rank.gather_f32(&mut wire, &self.candidate_offsets)?; Ok(wire)
        })?;
        Ok((wires[0][1] == 1.0).then(|| wires[0][0].to_bits()))
    }

    /// One layer traversal for independent cached tokens. HC and routed/shared
    /// experts reuse the prefill tile kernels, preserving each row's reduction
    /// order. GDN/QSA projections share the tile; recurrent
    /// and KV updates retain each session's one-token arithmetic and state.
    pub fn forward_batch(b: &Backend<'a>, batch: &mut DecodeBatch<'a, 'w>, sessions: &mut [&mut Self],
        tokens: &[u32], save_logits: &[bool]) -> Result<Vec<Result<Output>>> {
        let active = RoutingVerdict::slots(sessions.len())?;
        Self::forward_batch_survivors(b, batch, sessions, tokens, save_logits, active)
    }
    /// Keep the cohort's row slots and graph geometry after a rejection. The
    /// caller retains each retired session as padding and clears its mask bit;
    /// it may never re-admit that invalid session. Inactive rows neither update
    /// lexical/KV/recurrent histories nor commit outputs. Rank masks are agreed
    /// in the existing output gather, without an extra exchange.
    pub fn forward_batch_survivors(b: &Backend<'a>, batch: &mut DecodeBatch<'a, 'w>, sessions: &mut [&mut Self],
        tokens: &[u32], save_logits: &[bool], active: u16) -> Result<Vec<Result<Output>>> {
        if sessions.len() <= 1 || !sessions[0].ranks.first().is_some_and(Exchange::collective_profiling) {
            return Self::forward_batch_inner(b, batch, sessions, tokens, save_logits, active);
        }
        let count = sessions.len();
        for rank in &mut sessions[0].ranks { rank.profile_begin_step("decode_batch", count); }
        let result = Self::forward_batch_inner(b, batch, sessions, tokens, save_logits, active);
        for rank in &mut sessions[0].ranks { rank.profile_end_step(result.is_ok(), b.allocated_bytes()); }
        result
    }
    fn forward_batch_inner(b: &Backend<'a>, batch: &mut DecodeBatch<'a, 'w>, sessions: &mut [&mut Self],
        tokens: &[u32], save_logits: &[bool], active: u16) -> Result<Vec<Result<Output>>> {
        batch.last_collectives = None;
        let count = sessions.len();
        if count == 0 || count > batch.capacity || tokens.len() != count || save_logits.len() != count {
            return Err("invalid Flash decode batch shape".into());
        }
        RoutingVerdict::validate_survivors(count, active, batch.model.mtp_depth())?;
        if count == 1 {
            if !std::ptr::eq(batch.model, sessions[0].model) { return Err("Flash decode batch model differs".into()); }
            if active == 0 { return Ok(vec![Err(ROUTING_ERROR.into())]); }
            return sessions[0].forward(b, tokens, save_logits[0]).map(|output| vec![Ok(output)]);
        }
        // Batch reductions use only the lead session's tile-sized communication scratch.
        if count > sessions[0].tile {
            return Err(format!("Flash decode batch has {count} rows, exceeding lead session communication scratch tile capacity {}", sessions[0].tile));
        }
        let model = sessions[0].model;
        if !std::ptr::eq(batch.model, model) || sessions.iter().enumerate().any(|(i, s)| !std::ptr::eq(s.model, model) || s.ranks.len() != 1
            || active & (1 << i) != 0 && !s.valid || s.position == 0 || s.position >= s.capacity)
            || tokens.iter().any(|&t| t as usize >= VOCABULARY) {
            return Err("Flash decode batch needs independent cached sessions of the same distributed model".into());
        }
        for (i, session) in sessions.iter_mut().enumerate() {
            if active & (1 << i) != 0 { session.commit_verification(b, session.position)?; }
        }
        for s in sessions.iter_mut() { s.valid = false; }
        let exchanges = sessions[0].ranks[0].profile_stats().map(|stats| stats.exchanges);
        b.set_spin_waits(true);
        let result = (|| {
            let start = Instant::now();
            let mut load_ms = 0.0;
            let layers = model.checkpoint.config.layers.len();
            let graphs = model.resident && count <= batch.prepared_rows;
            let compact = model.persistent[0].plan.expert_output_rows.is_some();
            let segments = if compact { 4 } else { 3 };
            if count > batch.prepared_rows {
                // A wider first pass grows grouped expert scratch eagerly. Old
                // graphs must go before that allocation can be replaced.
                b.synchronize()?;
                for graph in &mut batch.graphs { *graph = None; }
                batch.graph_widths = [0; 2];
                batch.prepared_rows = count;
            }
            if graphs { batch.retain_graph_width(b, count)?; }
            batch.routing.clear(b, count, model.persistent[0].plan.rank as usize)?;
            if active != RoutingVerdict::slots(count)? { batch.routing.pad_rows(b, count, active)?; }
            batch.moe.reserve_grouped(b, count, model.main_activation_quantization(), compact.then_some(model.persistent[0].plan.ranks as usize))?;
            batch.moe.set_routing_flag(&batch.routing)?;
            batch.input.resize_within_capacity(count * H)?;
            batch.moe.resize(count)?;
            let mut embeddings = Vec::with_capacity(count);
            let mut rows = sessions.iter_mut().enumerate().map(|(i, s)| {
                let history = s.rows[0].history();
                let row = s.rows[0].capture_device(&model.persistent[0], tokens[i],
                    batch.row_wire.device_address() + (i * (EMBEDDING_BYTES + LEXICAL_BYTES)) as u64);
                if active & (1 << i) == 0 { s.rows[0].restore_history(history); }
                row
            }).collect::<Result<Vec<_>>>()?;
            RowBatch::gather_device(b, &mut rows, model.persistent[0].plan.rank as usize,
                &mut batch.row_wire, &mut batch.row_copies, |v, o| sessions[0].ranks[0].gather_bytes(v, o))?;
            for (i, s) in sessions.iter_mut().enumerate() {
                if active & (1 << i) == 0 {
                    b.zero_vector_range(&mut batch.hidden, i * W..(i + 1) * W)?;
                    embeddings.push(Vec::new());
                    continue;
                }
                let token = rows[i].token(0)?;
                s.front[0].expand(b, &token)?;
                embeddings.push(if s.mtp.is_empty() { Vec::new() } else { token.embedding.to_vec() });
                b.copy_range(&s.front[0].residual, &mut batch.hidden, 0, i * W, W)?;
                s.input[0].resize_within_capacity(H)?;
            }
            for layer in 0..model.checkpoint.config.layers.len() {
                let prefix = format!("model.language_model.layers.{layer}.");
                let streamed = if model.resident { None } else {
                    let loading = Instant::now();
                    let weights = Weights::load_selected(b, &model.root, &model.checkpoint,
                        model.persistent[0].plan.clone(), |name| name.starts_with(&prefix) && !name.starts_with(PLE))?;
                    load_ms += loading.elapsed().as_secs_f64() * 1000.0;
                    Some(weights)
                };
                let weights = streamed.as_ref().unwrap_or(&model.persistent[0]);
                // A restored row may not have gathered this layer's norms yet; every rank
                // gathers them here in the same row order, outside batch graphs.
                for s in sessions.iter_mut() { s.prepare_attention(b, std::slice::from_ref(weights), layer)?; }
                let connection = Hc::load(weights, &format!("{prefix}attn_hyper_connection"))?;
                if model.checkpoint.config.ple_layers.contains(&(layer as u64)) {
                    // One pass over the lexical weights projects every row (each
                    // with its solo sums); each session's convolution history
                    // then finishes its own row.
                    let (lead, rest) = sessions.split_first_mut().unwrap();
                    lead.ple[0].set_lexical(b, &lead.front[0].lexical, 0)?;
                    for (i, s) in rest.iter().enumerate() { lead.ple[0].set_lexical(b, &s.front[0].lexical, i + 1)?; }
                    let (keys, values) = lead.ple[0].project_lexical(b, count)?;
                    for (i, s) in sessions.iter_mut().enumerate() {
                        if active & (1 << i) == 0 { continue; }
                        s.ple[0].finish_rows(b, batch.hidden.device_address() + (i * W * 4) as u64,
                            keys + (i * W * 4) as u64, values + (i * H * 4) as u64, 1)?;
                    }
                }
                batch.segment(b, graphs.then_some(((count - 1) * layers + layer) * segments), |batch|
                    connection.mix_tiled(b, &mut batch.hc, &batch.hidden, &batch.input, count))?;
                if matches!(model.checkpoint.config.layers[layer], LayerKind::DeltaNet) {
                    let attention = sessions.iter_mut().map(|s| {
                        let state = s.gdn_cache[0][layer].take().ok_or("missing cached Flash GDN state")?;
                        Gdn::load_cached(b, weights, layer, Some(state))
                    }).collect::<Result<Vec<_>>>()?;
                    let rows = sessions.iter().zip(&attention).enumerate().map(|(i, (s, q))| {
                        let mut row = q.session_row(s.gdn[0].as_ref().ok_or("missing cached Flash GDN scratch")?,
                            s.gdn_norms[0][layer].as_ref().ok_or("missing cached Flash GDN norm")?, i)?;
                        row.padding = u32::from(active & (1 << i) == 0);
                        Ok(row)
                    }).collect::<Result<Vec<_>>>()?;
                    batch.gdn.upload_sessions(b, &rows)?;
                    batch.segment(b, graphs.then_some(((count - 1) * layers + layer) * segments + 1), |batch| {
                        attention[0].project_batch(b, &batch.gdn, &batch.input, count, false)?;
                        attention[0].step_batch(b, &batch.gdn, count)?;
                        attention[0].finish_batch(b, &batch.gdn, &mut batch.input, count)
                    })?;
                    for (s, q) in sessions.iter_mut().zip(attention) { s.gdn_cache[0][layer] = Some(q.into_state()); }
                } else {
                    let mut attention = sessions.iter_mut().map(|s| {
                        let state = s.qsa_cache[0][layer].take().ok_or("missing cached Flash QSA state")?;
                        Qsa::load_cached(b, weights, layer, s.capacity, Some(state))
                    }).collect::<Result<Vec<_>>>()?;
                    if active == RoutingVerdict::slots(count)? {
                        let rows = sessions.iter_mut().zip(&attention).map(|(s, q)| {
                            s.qsa[0].use_norm(s.qsa_norms[0][layer].as_ref().ok_or("missing cached Flash QSA norm")?)?;
                            q.session_row(&s.qsa[0])
                        }).collect::<Result<Vec<_>>>()?;
                        let shape = batch.qsa.upload_sessions(b, &rows)?;
                        let slot = (count - 1) * layers + layer;
                        let key = slot * segments + 1;
                        if batch.qsa_shapes[slot] != shape {
                            // Other widths/layers retain their own captured shape.
                            if batch.graphs[key].is_some() { b.synchronize()?; batch.graphs[key] = None; }
                            batch.qsa_shapes[slot] = shape;
                        }
                        batch.segment(b, graphs.then_some(key), |batch| {
                            attention[0].project_batch(b, &mut batch.qsa, &batch.input, count)?;
                            attention[0].attend_batch(b, &batch.qsa, count)?;
                            attention[0].finish_batch(b, &batch.qsa, &mut batch.input, count)
                        })?;
                        for q in &mut attention { q.commit_batch(); }
                    } else {
                        // Shared projections keep every slot; only survivors
                        // enter the existing solo kernels and advance history.
                        attention[0].project_batch(b, &mut batch.qsa, &batch.input, count)?;
                        for (i, (s, q)) in sessions.iter_mut().zip(&mut attention).enumerate() {
                            if active & (1 << i) == 0 { q.pad_batch_row(b, &batch.qsa, i)?; continue; }
                            s.qsa[0].use_norm(s.qsa_norms[0][layer].as_ref().ok_or("missing cached Flash QSA norm")?)?;
                            q.attend_batch_row(b, &batch.qsa, i, &mut s.qsa[0], 1)?;
                        }
                        attention[0].finish_batch(b, &batch.qsa, &mut batch.input, count)?;
                    }
                    for (s, q) in sessions.iter_mut().zip(attention) { s.qsa_cache[0][layer] = Some(q.into_state()); }
                }
                let lead = &mut *sessions[0];
                reduce_device(b, &mut lead.ranks, &mut lead.communication, None, &mut [&mut batch.input])?;
                let streamed_moe = if model.resident { None } else { Some(Moe::load(weights, layer, model.activation_quantization)?) };
                let mlp = if let Some(mlp) = &streamed_moe { mlp } else { &lead.resident_moe[0][layer] };
                batch.segment(b, graphs.then_some(((count - 1) * layers + layer) * segments + 2), |batch| {
                    connection.combine_tiled(b, &mut batch.hc, &mut batch.hidden, &batch.input, count)?;
                    mlp.project_batch(b, &mut batch.moe, &batch.hidden, count, batch.descriptors.get(layer))
                })?;
                if compact {
                    mlp.prepare_exchange(&mut batch.moe, count, batch.descriptors.get(layer))?;
                    exchange_middle(b, &mut lead.ranks[0], lead.mailbox.as_deref(), &batch.moe,
                        model.persistent[0].plan.rank as usize, count, false, &mut lead.communication[0])?;
                    batch.segment(b, graphs.then_some(((count - 1) * layers + layer) * segments + 3), |batch|
                        mlp.complete_middle(b, &batch.moe, count))?;
                    exchange_middle(b, &mut lead.ranks[0], lead.mailbox.as_deref(), &batch.moe,
                        model.persistent[0].plan.rank as usize, count, true, &mut lead.communication[0])?;
                } else {
                    reduce_device(b, &mut lead.ranks, &mut lead.communication, None, &mut [batch.moe.partials_mut()])?;
                }
                mlp.finish_batch(b, &mut batch.moe, &mut batch.hidden, count)?;
                let active = [streamed.as_ref().map_or(0, Weights::resident_bytes), 0];
                for s in sessions.iter_mut() { s.account(b, active, batch.device_bytes()); }
            }
            Hc::mixer(&model.persistent[0], MIXER)?.mix_tiled(b, &mut batch.hc, &batch.hidden, &batch.input, count)?;
            Head::project_batch(b, &mut sessions.iter_mut().map(|s| &mut s.head[0]).collect::<Vec<_>>(),
                &batch.input, &mut batch.logits)?;
            for (i, s) in sessions.iter_mut().enumerate() {
                if active & (1 << i) == 0 { continue; }
                b.copy_range(&batch.hidden, &mut s.hidden[0], i * W, 0, W)?;
                if !s.mtp.is_empty() {
                    let hidden = [s.hidden[0].slice(0..W)?];
                    s.observe_mtp(b, s.position, 1, &hidden, std::slice::from_ref(&embeddings[i]), 0)?;
                }
            }
            let mut outputs = Self::collect_batch_output(b, batch, sessions, save_logits, active)?.into_iter()
                .map(|row| row.map(|(token, logits)| Output { token, logits, elapsed_ms: 0.0, load_ms })).collect::<Vec<_>>();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            for ((s, output), &token) in sessions.iter_mut().zip(&mut outputs).zip(tokens) {
                let Ok(output) = output else { continue; };
                if s.caching { s.prefill_tail.push(token); }
                s.position += 1;
                s.valid = true;
                output.elapsed_ms = elapsed_ms;
            }
            Ok(outputs)
        })();
        batch.last_collectives = exchanges.zip(sessions[0].ranks[0].profile_stats())
            .map(|(before, after)| after.exchanges - before);
        b.set_spin_waits(false);
        result
    }
}
