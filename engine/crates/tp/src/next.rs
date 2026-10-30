//! World-N Next session schedule, independent of the single-GPU kernel lane.
//! The native HTTP and capture entry points share this host-testable schedule.
//! The GPU adapter supplies rank-local projections and recurrent/KV state;
//! transport gathers partials, then every rank folds them in ascending order.
use crate::Result;
use engine_model::flash::shard::{ShardCheckpoint, ShardPlan};
use std::sync::{Arc, Mutex};
use std::{net::SocketAddr, ops::Range, path::Path};

/// Serving retains the EP layout; experimental splits stay in native gates.
pub fn require_serving_split(value: Option<&str>) -> Result<()> {
    match value {
        None | Some("ep") => Ok(()),
        Some("ep2tp2") => Err("NEXT_TP_SPLIT=ep2tp2 is unqualified for serving; use ep".into()),
        _ => Err("NEXT_TP_SPLIT must be ep for serving".into()),
    }
}

/// One logical checkpoint name across world-one and preview rank services.
pub fn model_id(root: &Path) -> Result<String> {
    Ok(root
        .file_name()
        .ok_or("checkpoint needs a model name")?
        .to_string_lossy()
        .to_ascii_lowercase())
}

/// Explicit layout opt-in; read before loading weights or opening sessions.
pub fn ep2_enabled(world: usize) -> Result<bool> {
    let options = spark_engine::flash_next_options::Options::load()?;
    parse_ep2(Some(if options.fp4_ep { "1" } else { "0" }), world)
}
/// Legacy switch name retained for existing W2 launchers; W4 uses the same EP.
pub fn parse_ep2(value: Option<&str>, world: usize) -> Result<bool> {
    let enabled = match value {
        Some("1") => true,
        Some("0") | None => false,
        _ => return Err("VADE_FLASHNEXT_FP4_EP2 must be 0 or 1".into()),
    };
    if enabled && !matches!(world, 2 | 4) {
        return Err("VADE_FLASHNEXT_FP4_EP2 requires world two or four".into());
    }
    Ok(enabled)
}

/// Parse before checkpoint I/O, transport creation or CUDA work on every host.
pub struct RankArgs<'a> {
    pub rank: usize,
    pub world: usize,
    pub listen: SocketAddr,
    pub next: SocketAddr,
    pub bind: Option<SocketAddr>,
    pub requests: usize,
    pub allow_remote: bool,
    pub key: Option<&'a Path>,
}
impl<'a> RankArgs<'a> {
    pub fn parse(a: &'a [String], http: bool) -> Result<Self> {
        if a.len() < 9 || !http && a.len() != 9 {
            return Err("expected RANK WORLD LISTEN NEXT PTX MODEL TENSOR_MAP followed by BIND REQUESTS or CASES NEW_OUTPUT_DIR".into());
        }
        let rank = a[0].parse().map_err(|_| "rank")?;
        let world = a[1].parse().map_err(|_| "world")?;
        let supported: &[usize] = if http { &[2, 4] } else { &[1, 2, 4] };
        if !supported.contains(&world) || rank >= world {
            return Err("Next requires rank < world in 1/2/4; HTTP requires W2/W4".into());
        }
        let listen: SocketAddr = a[2].parse().map_err(|_| "listen")?;
        let next: SocketAddr = a[3].parse().map_err(|_| "next")?;
        if [listen, next].iter().any(|a| matches!(a.port(), 0 | 65535)) {
            return Err("control ports must leave room for the data port".into());
        }
        let mut out = Self {
            rank,
            world,
            listen,
            next,
            bind: None,
            requests: 0,
            allow_remote: false,
            key: None,
        };
        if http {
            let bind: SocketAddr = a[7].parse().map_err(|_| "bind")?;
            if bind.port() == 0 {
                return Err("HTTP bind port must be nonzero".into());
            }
            out.bind = Some(bind);
            out.requests = a[8].parse().map_err(|_| "requests")?;
            let mut options = a[9..].iter();
            while let Some(option) = options.next() {
                match option.as_str() {
                    "--allow-remote" if !out.allow_remote => out.allow_remote = true,
                    "--api-key-file" if out.key.is_none() => {
                        out.key = Some(Path::new(options.next().ok_or("key path missing")?))
                    }
                    _ => return Err("unknown or duplicate option".into()),
                }
            }
            if !bind.ip().is_loopback() && !out.allow_remote {
                return Err("remote bind requires --allow-remote".into());
            }
        }
        Ok(out)
    }
}

/// Loader contract: F8 PLE scales are disk metadata, never matrices.
/// Kept host-visible so the exact production predicate can be tested on a Mac.
pub fn native_projection<'n>(
    name: &'n str,
    t: &engine_model::flash::shard::TensorShard,
    q: &engine_model::flash::mixed::Mixed,
) -> Option<&'n str> {
    use engine_model::{
        flash::{mixed::Storage, shard::Placement},
        safetensors::Dtype,
        shapes::Store,
    };
    let module = name.strip_suffix(".weight")?;
    (t.placement == Placement::Resident
        && t.role.store == Store::Weight
        && t.tensor.dtype == Dtype::F8E4M3
        && t.bytes() != 0
        && module.starts_with("model.language_model.")
        && q.storage(module) == Storage::Mxfp8)
        .then_some(module)
}

/// The controller can interleave independent sessions on one ordered transport.
/// A lock covers a whole collective; controller command order still determines
/// which session every worker advances next.
pub struct SharedTransport<T> {
    inner: Arc<Mutex<T>>,
    rank: usize,
    world: usize,
}
impl<T: RankTransport> SharedTransport<T> {
    pub fn new(inner: T) -> Self {
        Self {
            rank: inner.rank(),
            world: inner.world(),
            inner: Arc::new(Mutex::new(inner)),
        }
    }
}
impl<T> Clone for SharedTransport<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            rank: self.rank,
            world: self.world,
        }
    }
}
impl<T: RankTransport> RankTransport for SharedTransport<T> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world(&self) -> usize {
        self.world
    }
    fn bytes(&mut self, values: &mut [u8], row: usize) -> Result<()> {
        let timer = crate::next_timing::start();
        let result = self
            .inner
            .lock()
            .map_err(|_| "Next transport poisoned")?
            .bytes(values, row);
        crate::next_timing::data(timer);
        result
    }
    fn floats(&mut self, values: &mut [f32], row: usize) -> Result<()> {
        let timer = crate::next_timing::start();
        let result = self
            .inner
            .lock()
            .map_err(|_| "Next transport poisoned")?
            .floats(values, row);
        crate::next_timing::data(timer);
        result
    }
}

/// Both the existing TP control transport and Flash's seqpar data transport
/// implement the same rank-major gather contract. No new sockets or RDMA code.
pub trait RankTransport {
    fn rank(&self) -> usize;
    fn world(&self) -> usize;
    fn bytes(&mut self, values: &mut [u8], row_bytes: usize) -> Result<()>;
    fn floats(&mut self, values: &mut [f32], row_floats: usize) -> Result<()>;
}
impl RankTransport for crate::transport::Ring {
    fn rank(&self) -> usize {
        self.rank()
    }
    fn world(&self) -> usize {
        self.world()
    }
    fn bytes(&mut self, v: &mut [u8], row: usize) -> Result<()> {
        self.all_gather(v, row)
    }
    fn floats(&mut self, v: &mut [f32], row: usize) -> Result<()> {
        // Opaque transfer only; all arithmetic happens after rank ordering.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };
        self.all_gather(bytes, row * 4)
    }
}
impl RankTransport for spark_seqpar::Ring {
    fn rank(&self) -> usize {
        self.rank()
    }
    fn world(&self) -> usize {
        self.world()
    }
    fn bytes(&mut self, v: &mut [u8], row: usize) -> Result<()> {
        self.all_gather_bytes(v, &(0..=self.world()).map(|r| r * row).collect::<Vec<_>>())
    }
    fn floats(&mut self, v: &mut [f32], row: usize) -> Result<()> {
        self.all_gather_f32(v, &(0..=self.world()).map(|r| r * row).collect::<Vec<_>>())
    }
}

/// Bytes of one rank's block per data collective (the rank service's RDMA mailbox slot).
pub const RANK_MAILBOX_SLOT: usize = 1 << 20;
/// `v.iter().all(|x| x.is_finite())` without the per-element early exit: one OR of exponent
/// tests per word, so the host fold of a prompt chunk vectorizes (same verdict).
pub(crate) fn all_finite(v: &[f32]) -> bool {
    v.iter().fold(0u32, |bad, x| bad | u32::from(x.to_bits() & 0x7f80_0000 == 0x7f80_0000)) == 0
}
/// The same first + second addition as copy/add, with its overflow check.
fn sum_pair(left: &[f32], right: &[f32], out: &mut [f32]) -> bool {
    debug_assert_eq!(left.len(), out.len());
    debug_assert_eq!(right.len(), out.len());
    out.iter_mut().zip(left).zip(right).fold(0u32, |bad, ((value, &a), &b)| {
        *value = a + b;
        bad | u32::from(value.to_bits() & 0x7f80_0000 == 0x7f80_0000)
    }) == 0
}
/// Ordered row fold: fuse the first copy/add, then visit any remaining rows in
/// their original order. Finite inputs cannot recover from an intermediate
/// overflow, so OR-ing the per-pass checks gives the final overflow verdict.
fn sum_rows<'a>(mut rows: impl Iterator<Item = &'a [f32]>, out: &mut [f32]) -> bool {
    let first = rows.next().expect("rank fold needs one row");
    let Some(second) = rows.next() else {
        out.copy_from_slice(first);
        return all_finite(out);
    };
    let mut finite = sum_pair(first, second, out);
    for row in rows {
        let bad = out.iter_mut().zip(row).fold(0u32, |bad, (value, &addend)| {
            *value += addend;
            bad | u32::from(value.to_bits() & 0x7f80_0000 == 0x7f80_0000)
        });
        finite &= bad == 0;
    }
    finite
}
/// Single-token and batched kernel callbacks must keep
/// the same per-row arithmetic and causal state order (implemented by the row kernels).
/// Every session owns a separate K: no GDN/KV/PLE state lives in NextModel.
pub trait RankKernels {
    /// Local row contributions: H embedding floats followed by H lexical floats.
    /// Non-owned rows are zeros. PLE history advances exactly once per token.
    fn front(
        &mut self,
        checkpoint: &ShardCheckpoint,
        token: u32,
        position: usize,
        rows: &mut [f32],
    ) -> Result<()>;
    /// Expand all HC streams and retain the gathered lexical row for its layer.
    fn embed(&mut self, rows: &[f32]) -> Result<()>;
    /// HC mix, local QSA/GDN projection/state, and local output-projection sum.
    /// No cross-rank state exchange: only the H-wide output is communicated.
    fn attention(&mut self, layer: usize, position: usize, partial: &mut [f32]) -> Result<()>;
    fn attention_rows(rows: &mut [(&mut Self, usize, &mut [f32])], layer: usize) -> Vec<Result<()>>
    where
        Self: Sized,
    {
        rows.iter_mut()
            .map(|(k, pos, out)| k.attention(layer, *pos, out))
            .collect()
    }
    fn inject_attention(&mut self, layer: usize, reduced: &[f32]) -> Result<()>;
    /// HC mix + identical deterministic router on every rank; selected OWNED
    /// experts in ascending ID, then the gated shared TP partial, into H floats.
    fn moe(&mut self, layer: usize, partial: &mut [f32]) -> Result<()>;
    fn moe_rows(rows: &mut [(&mut Self, &mut [f32])], layer: usize) -> Vec<Result<()>>
    where
        Self: Sized,
    {
        rows.iter_mut().map(|(k, out)| k.moe(layer, out)).collect()
    }
    /// Selected global IDs and weights, sorted by ID. Piggybacked on the MoE
    /// gather so disagreeing routes cannot commit a token on any rank.
    fn separate_shared(&self) -> bool {
        false
    }
    /// EP adds the replicated shared expert AFTER the rank 0..world fold. Never put
    /// it in both partials, or fuse this addition into HC/residual/normalization.
    fn add_replicated_shared(&mut self, _reduced: &mut [f32]) -> Result<()> {
        Ok(())
    }
    fn routing(&self) -> &[(u32, f32)];
    fn inject_moe(&mut self, layer: usize, reduced: &[f32]) -> Result<()>;
    /// Final replicated HC mixer and the owned vocabulary rows of lm_head.
    fn logits(&mut self, vocabulary: Range<u64>, local: &mut [f32]) -> Result<()>;
    /// Reset every recurrence, convolution, KV/index cursor and PLE history.
    fn reset(&mut self) -> Result<()>;
    /// Prompt rows one chunk may hold; 1 = no chunk form (prompt rows run one at a time).
    /// The chunk hooks below run `rows` consecutive prompt tokens of this session layer by
    /// layer with each row's single-token arithmetic and causal order; buffers are row-major.
    fn chunk_rows(&self) -> usize {
        1
    }
    fn embed_chunk(&mut self, _rows: &[f32], _tokens: usize) -> Result<()> {
        Err("no chunk form".into())
    }
    fn attention_chunk(&mut self, _layer: usize, _out: &mut [f32], _tokens: usize) -> Result<()> {
        Err("no chunk form".into())
    }
    fn inject_attention_chunk(&mut self, _layer: usize, _reduced: &[f32], _tokens: usize) -> Result<()> {
        Err("no chunk form".into())
    }
    /// Rows of `width` floats (routed partial, then the shared TP partial when separate).
    fn moe_chunk(&mut self, _layer: usize, _out: &mut [f32], _tokens: usize) -> Result<()> {
        Err("no chunk form".into())
    }
    /// Row `row`'s selected global IDs and weights, sorted by ID.
    fn chunk_routing(&self, _row: usize) -> &[(u32, f32)] {
        &[]
    }
    fn inject_moe_chunk(&mut self, _layer: usize, _reduced: &[f32], _tokens: usize) -> Result<()> {
        Err("no chunk form".into())
    }
    /// After the last layer: the final row becomes the single-token state `logits` reads.
    fn finish_chunk(&mut self, _tokens: usize) -> Result<()> {
        Err("no chunk form".into())
    }
    /// Cohort form (one decode row per session, row r = `rows[r]`'s session): `rows[0]` runs the
    /// row-batched chunk hooks above on its chunk buffers; only the attention (whose causal
    /// state is per session) and the final hand-back need every row's kernels. 1 = none.
    fn cohort_rows(&self) -> usize {
        1
    }
    fn cohort_attention(_rows: &mut [&mut Self], _layer: usize, _out: &mut [f32]) -> Result<()>
    where
        Self: Sized,
    {
        Err("no cohort form".into())
    }
    /// Row r's final residual becomes `rows[r]`'s single-token state for its head.
    fn cohort_finish(_rows: &mut [&mut Self]) -> Result<()>
    where
        Self: Sized,
    {
        Err("no cohort form".into())
    }
}

/// Weights are immutable and shared by sessions; the loader only streams this
/// rank's resident selections. It never instantiates a world-1 GPU NextModel.
pub struct NextModel {
    pub checkpoint: ShardCheckpoint,
    identity: u64,
}
impl NextModel {
    pub fn load(root: &Path, tensor_map: &Path, rank: usize, world: usize) -> Result<Self> {
        Self::from_checkpoint(
            ShardCheckpoint::open(root, tensor_map, rank, world).map_err(|e| e.to_string())?,
        )
    }
    pub fn from_checkpoint(checkpoint: ShardCheckpoint) -> Result<Self> {
        // Agreement fingerprint, not a payload hash. Existing startup's
        // verify_artifacts remains required before loading on real ranks.
        let mut identity = 0xcbf29ce484222325u64;
        let mut hash = |bytes: &[u8]| {
            for &b in bytes {
                identity = (identity ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        };
        hash(
            format!(
                "{:?} expert_tp={}",
                checkpoint.plan.shape, checkpoint.plan.expert_tp
            )
            .as_bytes(),
        );
        if checkpoint.plan.ep2 {
            hash(format!("replicated-ep-rank-order-world{}", checkpoint.plan.world).as_bytes());
        }
        for (name, tensor) in &checkpoint.plan.tensors {
            hash(
                format!(
                    "{name}\0{}\0{:?}\0{:?}\0{:?}",
                    tensor.file, tensor.tensor.dtype, tensor.tensor.shape, tensor.role
                )
                .as_bytes(),
            );
        }
        Ok(Self {
            checkpoint,
            identity,
        })
    }
    /// Ranks call in controller command order with the same nonzero session ID.
    /// Opening collectively checks geometry, capacity and kernel reset status.
    pub fn session<T: RankTransport, K: RankKernels>(
        &self,
        transport: T,
        kernels: K,
        id: u64,
        capacity: usize,
    ) -> Result<NextSession<'_, T, K>> {
        let plan = &self.checkpoint.plan;
        if transport.rank() != plan.rank || transport.world() != plan.world {
            return Err("Next transport/checkpoint rank or world mismatch".into());
        }
        let h = usize::try_from(plan.shape.hidden).map_err(|_| "Next hidden size overflow")?;
        let top = plan
            .shape
            .layers
            .iter()
            .map(|l| match l.ffn {
                engine_model::shapes::Ffn::Experts(e) => e.selected as usize,
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        let wire = (2 * h + 2 * top + 1).max(4) * plan.world;
        let mut session = NextSession {
            model: self,
            transport,
            kernels,
            id,
            capacity,
            position: 0,
            failed: false,
            epoch: 0,
            wire: vec![0.; wire],
            local: vec![0.; 2 * h],
            reduced: vec![0.; 2 * h],
            logits: vec![0.; (plan.vocabulary().end - plan.vocabulary().start) as usize],
            header: vec![0; 64 * plan.world],
        };
        let reset = session.kernels.reset();
        let valid =
            reset.is_ok() && id != 0 && capacity > 0 && capacity as u64 <= plan.shape.max_positions;
        session.agree(0, 0, valid)?;
        Ok(session)
    }
}

/// A failed partial step is poisoned until a collective reset; callers cannot
/// accidentally reuse half-updated GDN/KV state. Request validation is agreed
/// before any state mutation, so a rejected token can be retried safely.
pub struct NextSession<'m, T, K> {
    model: &'m NextModel,
    transport: T,
    kernels: K,
    id: u64,
    capacity: usize,
    position: usize,
    epoch: u64,
    failed: bool,
    wire: Vec<f32>,
    local: Vec<f32>,
    reduced: Vec<f32>,
    logits: Vec<f32>,
    header: Vec<u8>,
}
impl<T: RankTransport, K: RankKernels> NextSession<'_, T, K> {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn is_failed(&self) -> bool {
        self.failed
    }
    pub fn kernels(&self) -> &K {
        &self.kernels
    }
    pub fn transport(&self) -> &T {
        &self.transport
    }
    pub fn plan(&self) -> &ShardPlan {
        &self.model.checkpoint.plan
    }
    fn agree(&mut self, op: u64, token: u32, valid: bool) -> Result<()> {
        let rank = self.transport.rank();
        let words = [
            op,
            self.id,
            self.epoch,
            self.position as u64,
            token as u64,
            self.capacity as u64,
            self.model.identity,
            u64::from(valid),
        ];
        for (dst, word) in self.header[rank * 64..(rank + 1) * 64]
            .chunks_exact_mut(8)
            .zip(words)
        {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        self.transport
            .bytes(&mut self.header, 64)
            .inspect_err(|_| self.failed = true)?;
        if self
            .header
            .chunks_exact(64)
            .any(|row| row != &self.header[..64])
        {
            self.failed = true;
            return Err(
                "Next ranks disagree on session, token, position, geometry or request validity"
                    .into(),
            );
        }
        if !valid {
            return Err("invalid Next session request".into());
        }
        Ok(())
    }
    /// Each payload has a status float. Even a local kernel error participates
    /// in the scheduled gather so peers fail together instead of deadlocking.
    /// Rank-order reduction fixes bytes per world size, not across world sizes:
    /// different topologies regroup expert partials and have separate goldens.
    fn sum(&mut self, width: usize, status: Result<()>, route: Option<(usize, u64)>) -> Result<()> {
        let row = self.prepare_sum(width, &status, route);
        self.transport
            .floats(&mut self.wire[..row * self.transport.world()], row)?;
        self.finish_sum(width, row, status)
    }
    fn prepare_sum(
        &mut self,
        width: usize,
        status: &Result<()>,
        route: Option<(usize, u64)>,
    ) -> usize {
        let routes = route.map_or(&[][..], |_| self.kernels.routing());
        let extra = route.map_or(0, |(top, _)| top * 2);
        let (rank, world, row) = (
            self.transport.rank(),
            self.transport.world(),
            width + 1 + extra,
        );
        let valid_route = route.is_none_or(|(top, count)| {
            routes.len() == top
                && routes.iter().all(|&(id, weight)| {
                    u64::from(id) < count && weight.is_finite() && weight >= 0.
                })
                && routes.windows(2).all(|pair| pair[0].0 < pair[1].0)
        });
        let valid =
            status.is_ok() && valid_route && self.local[..width].iter().all(|v| v.is_finite());
        let wire = &mut self.wire[..row * world];
        wire[rank * row..(rank + 1) * row].fill(0.);
        wire[rank * row] = if valid { 1. } else { 0. };
        wire[rank * row + 1..rank * row + 1 + width].copy_from_slice(&self.local[..width]);
        if valid {
            for (dst, &(id, weight)) in wire[rank * row + 1 + width..(rank + 1) * row]
                .chunks_exact_mut(2)
                .zip(routes)
            {
                dst.copy_from_slice(&[id as f32, weight]);
            }
        }
        row
    }
    fn finish_sum(&mut self, width: usize, row: usize, status: Result<()>) -> Result<()> {
        let wire = &self.wire[..row * self.transport.world()];
        if wire
            .chunks_exact(row)
            .any(|v| v[0] != 1. || v[1..].iter().any(|x| !x.is_finite()))
        {
            return Err(status
                .err()
                .unwrap_or_else(|| "Next rank kernel failed or produced nonfinite output".into()));
        }
        if wire.chunks_exact(row).any(|peer| {
            peer[1 + width..]
                .iter()
                .zip(&wire[1 + width..row])
                .any(|(a, b)| a.to_bits() != b.to_bits())
        }) {
            return Err("Next rank routing mismatch".into());
        }
        // Bytes are fixed per world size: rank-order folding is deterministic
        // on the same topology. Different world sizes regroup expert partials
        // and use their own byte goldens; cross-world bytes are not promised.
        self.reduced[..width].copy_from_slice(&wire[1..1 + width]);
        for peer in wire[row..].chunks_exact(row) {
            for (value, &addend) in self.reduced[..width].iter_mut().zip(&peer[1..1 + width]) {
                *value += addend;
            }
        }
        if self.reduced[..width].iter().any(|x| !x.is_finite()) {
            return Err("Next reduction overflow".into());
        }
        Ok(())
    }
    pub fn step(&mut self, token: u32) -> Result<u32> {
        let timer = crate::next_timing::start();
        let valid = !self.failed
            && self.position < self.capacity
            && u64::from(token) < self.plan().shape.vocab;
        self.agree(1, token, valid)?;
        let result = self.step_inner(token);
        if result.is_err() {
            self.failed = true;
        }
        crate::next_timing::step(timer, self.transport.rank(), 1);
        result
    }
    /// A prompt row whose prediction nobody reads: the same agreement, layers,
    /// collectives and causal state as step, without the final HC mix, the
    /// vocabulary head and the argmax (every buffer that mix writes is
    /// rewritten before its next read). Ranks exchange only their status.
    pub fn advance(&mut self, token: u32) -> Result<()> {
        let timer = crate::next_timing::start();
        let valid = !self.failed
            && self.position < self.capacity
            && u64::from(token) < self.plan().shape.vocab;
        self.agree(3, token, valid)?;
        let result = self.advance_inner(token);
        if result.is_err() {
            self.failed = true;
        }
        crate::next_timing::step(timer, self.transport.rank(), 1);
        result
    }
    fn advance_inner(&mut self, token: u32) -> Result<()> {
        let status = self.layers(token)?;
        let row = 4;
        let world = self.transport.world();
        let rank = self.transport.rank();
        self.wire[rank * row..(rank + 1) * row]
            .copy_from_slice(&[if status.is_ok() { 1. } else { 0. }, 0., 0., 0.]);
        self.transport.floats(&mut self.wire[..row * world], row)?;
        if self.wire[..row * world]
            .chunks_exact(row)
            .any(|peer| peer[0] != 1. || peer.iter().any(|x| !x.is_finite()))
        {
            return Err("Next rank prompt step failed".into());
        }
        self.position += 1;
        Ok(())
    }
    fn step_inner(&mut self, token: u32) -> Result<u32> {
        let status = self.layers(token)?;
        self.prepare_head(status);
        let row = 4;
        let world = self.transport.world();
        self.transport.floats(&mut self.wire[..row * world], row)?;
        self.finish_head()
    }
    /// Embedding and every layer of one token; the outer error is a collective
    /// failure, the inner status a local one carried to the next exchange.
    fn layers(&mut self, token: u32) -> Result<Result<()>> {
        let h = self.plan().shape.hidden as usize;
        self.local.fill(0.);
        let status = self.kernels.front(
            &self.model.checkpoint,
            token,
            self.position,
            &mut self.local,
        );
        self.sum(2 * h, status, None)?;
        let mut status = self.kernels.embed(&self.reduced);
        for layer in 0..self.plan().shape.layers.len() {
            self.local[..h].fill(0.);
            status = status.and_then(|_| {
                self.kernels
                    .attention(layer, self.position, &mut self.local[..h])
            });
            if self.plan().ep2 {
                // Carry a local attention failure into the single MoE gather,
                // so every peer still executes the same collective schedule.
                self.reduced[..h].copy_from_slice(&self.local[..h]);
                if self.local[..h].iter().any(|x| !x.is_finite()) {
                    status = Err("EP2 nonfinite attention".into());
                }
            } else {
                self.sum(h, status.clone(), None)?;
            }
            status = status.and_then(|_| self.kernels.inject_attention(layer, &self.reduced[..h]));
            let width = if !self.plan().ep2 && self.kernels.separate_shared() {
                2 * h
            } else {
                h
            };
            self.local[..width].fill(0.);
            status = status.and_then(|_| self.kernels.moe(layer, &mut self.local[..width]));
            let engine_model::shapes::Ffn::Experts(experts) = self.plan().shape.layers[layer].ffn
            else {
                return Err("Next skeleton requires MoE layers".into());
            };
            self.sum(
                width,
                status,
                Some((experts.selected as usize, experts.count)),
            )?;
            if width == 2 * h {
                for i in 0..h {
                    self.reduced[i] += self.reduced[h + i];
                }
            }
            if self.plan().ep2 {
                self.kernels.add_replicated_shared(&mut self.reduced[..h])?;
            }
            status = self.kernels.inject_moe(layer, &self.reduced[..h]);
        }
        Ok(status)
    }
    fn prepare_head(&mut self, status: Result<()>) {
        let vocabulary = self.plan().vocabulary();
        let status = status.and_then(|_| self.kernels.logits(vocabulary.clone(), &mut self.logits));
        let mut best = (u32::MAX, 0.);
        let valid = status.is_ok() && self.logits.iter().all(|x| x.is_finite());
        for (index, &logit) in self.logits.iter().enumerate() {
            if best.0 == u32::MAX || logit > best.1 {
                best = ((vocabulary.start as usize + index) as u32, logit);
            }
        }
        // Numeric token IDs are exact for the model shape's MAX_VOCAB bound;
        // a distinct presence bit represents an empty vocabulary shard.
        let row = 4;
        let rank = self.transport.rank();
        self.wire[rank * row..(rank + 1) * row].copy_from_slice(&[
            if valid { 1. } else { 0. },
            if self.logits.is_empty() { 0. } else { 1. },
            best.0.min(1 << 24) as f32,
            best.1,
        ]);
    }
    fn finish_head(&mut self) -> Result<u32> {
        let row = 4;
        let world = self.transport.world();
        let mut winner: Option<(u32, f32)> = None;
        for peer in self.wire[..row * world].chunks_exact(row) {
            if peer[0] != 1. || peer.iter().any(|x| !x.is_finite()) {
                return Err("Next rank head failed".into());
            }
            if peer[1] == 0. {
                continue;
            }
            let token = peer[2] as u32;
            if token as u64 >= self.plan().shape.vocab {
                return Err("Next candidate outside vocabulary".into());
            }
            if winner.is_none_or(|(id, logit)| peer[3] > logit || peer[3] == logit && token < id) {
                winner = Some((token, peer[3]));
            }
        }
        let winner = winner.ok_or("empty Next vocabulary")?.0;
        self.position += 1;
        Ok(winner)
    }
    /// One rank-major gather per existing boundary for the entire cohort.
    /// Every row retains the single-request arithmetic and ascending rank fold.
    pub fn step_rows(sessions: &mut [&mut Self], tokens: &[u32]) -> Result<Vec<u32>> {
        if sessions.is_empty() || sessions.len() > 8 || sessions.len() != tokens.len() {
            return Err("invalid Next rows".into());
        }
        let model = sessions[0].model;
        if sessions.iter().enumerate().any(|(i, s)| {
            !std::ptr::eq(s.model, model) || sessions[..i].iter().any(|p| p.id == s.id)
        }) {
            return Err("Next rows require distinct sessions of one model".into());
        }
        let timer = crate::next_timing::start();
        // Agree on the full ordered cohort before mutating any causal state.
        for (s, &t) in sessions.iter_mut().zip(tokens) {
            s.agree(
                3 + 16 * tokens.len() as u64,
                t,
                !s.failed
                    && s.position < s.capacity
                    && u64::from(t) < model.checkpoint.plan.shape.vocab,
            )?;
        }
        let result = Self::rows_inner(sessions, tokens);
        if result.is_err() {
            for s in sessions.iter_mut() {
                s.failed = true;
            }
        }
        crate::next_timing::step(timer, sessions[0].transport.rank(), sessions.len());
        result
    }
    fn gather_rows(sessions: &mut [&mut Self], row: usize) -> Result<()> {
        let (rank, world, n) = (
            sessions[0].transport.rank(),
            sessions[0].transport.world(),
            sessions.len(),
        );
        let mut wire = vec![0.; row * n * world];
        for (i, s) in sessions.iter().enumerate() {
            wire[(rank * n + i) * row..(rank * n + i + 1) * row]
                .copy_from_slice(&s.wire[rank * row..(rank + 1) * row]);
        }
        sessions[0].transport.floats(&mut wire, row * n)?;
        for (i, s) in sessions.iter_mut().enumerate() {
            for r in 0..world {
                s.wire[r * row..(r + 1) * row]
                    .copy_from_slice(&wire[(r * n + i) * row..(r * n + i + 1) * row]);
            }
        }
        Ok(())
    }
    fn sum_rows(
        sessions: &mut [&mut Self],
        width: usize,
        status: Vec<Result<()>>,
        route: Option<(usize, u64)>,
    ) -> Result<()> {
        let mut row = 0;
        for (s, st) in sessions.iter_mut().zip(&status) {
            row = s.prepare_sum(width, st, route);
        }
        Self::gather_rows(sessions, row)?;
        for (s, st) in sessions.iter_mut().zip(status) {
            s.finish_sum(width, row, st)?;
        }
        Ok(())
    }
    /// rows_inner through the cohort form: every row keeps its one-token arithmetic and its
    /// session's causal state; one gather per boundary for the cohort, as rows_inner.
    fn cohort_inner(sessions: &mut [&mut Self], tokens: &[u32]) -> Result<Vec<u32>> {
        let model = sessions[0].model;
        let n = sessions.len();
        let h = model.checkpoint.plan.shape.hidden as usize;
        let mut local = vec![0f32; n * 2 * h];
        let mut status = Ok(());
        for (i, (s, &t)) in sessions.iter_mut().zip(tokens).enumerate() {
            let row = &mut local[i * 2 * h..(i + 1) * 2 * h];
            status = status.and(s.kernels.front(&model.checkpoint, t, s.position, row));
        }
        let mut reduced = vec![0f32; n * 2 * h];
        sessions[0].sum_chunk(&local, &mut reduced, 2 * h, n, status, None)?;
        let mut status = sessions[0].kernels.embed_chunk(&reduced, n);
        let separate = sessions[0].kernels.separate_shared();
        let width = if separate { 2 * h } else { h };
        let mut local = vec![0f32; n * width];
        for layer in 0..model.checkpoint.plan.shape.layers.len() {
            local[..n * h].fill(0.);
            status = status.and_then(|_| {
                let mut ks = sessions.iter_mut().map(|s| &mut s.kernels).collect::<Vec<_>>();
                K::cohort_attention(&mut ks, layer, &mut local[..n * h])
            });
            sessions[0].sum_chunk(&local[..n * h], &mut reduced[..n * h], h, n, status.clone(), None)?;
            status = status.and_then(|_| sessions[0].kernels.inject_attention_chunk(layer, &reduced[..n * h], n));
            local.fill(0.);
            status = status.and_then(|_| sessions[0].kernels.moe_chunk(layer, &mut local, n));
            let engine_model::shapes::Ffn::Experts(e) = model.checkpoint.plan.shape.layers[layer].ffn else {
                return Err("Next rows require MoE".into());
            };
            sessions[0].sum_chunk(&local, &mut reduced[..n * width], width, n, status, Some((e.selected as usize, e.count)))?;
            let mut block = vec![0f32; n * h];
            for (r, out) in block.chunks_exact_mut(h).enumerate() {
                let row = &reduced[r * width..(r + 1) * width];
                out.copy_from_slice(&row[..h]);
                if separate {
                    for i in 0..h {
                        out[i] += row[h + i];
                    }
                }
            }
            status = sessions[0].kernels.inject_moe_chunk(layer, &block, n);
        }
        status = status.and_then(|_| {
            let mut ks = sessions.iter_mut().map(|s| &mut s.kernels).collect::<Vec<_>>();
            K::cohort_finish(&mut ks)
        });
        for s in sessions.iter_mut() {
            s.prepare_head(status.clone());
        }
        Self::gather_rows(sessions, 4)?;
        sessions.iter_mut().map(|s| s.finish_head()).collect()
    }
    fn rows_inner(sessions: &mut [&mut Self], tokens: &[u32]) -> Result<Vec<u32>> {
        let model = sessions[0].model;
        if sessions.len() >= 2
            && !model.checkpoint.plan.ep2
            && sessions.iter().all(|s| s.kernels.cohort_rows() >= sessions.len())
        {
            return Self::cohort_inner(sessions, tokens);
        }
        let plan = &model.checkpoint.plan;
        let h = plan.shape.hidden as usize;
        let status = sessions
            .iter_mut()
            .zip(tokens)
            .map(|(s, &t)| {
                s.local.fill(0.);
                s.kernels
                    .front(&model.checkpoint, t, s.position, &mut s.local)
            })
            .collect();
        Self::sum_rows(sessions, 2 * h, status, None)?;
        let mut status = sessions
            .iter_mut()
            .map(|s| s.kernels.embed(&s.reduced))
            .collect::<Vec<_>>();
        spark_engine::flash_next_rows::layers(plan.shape.layers.len(), |li, attention| {
            if attention {
                let indices = status
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.is_ok().then_some(i))
                    .collect::<Vec<_>>();
                let mut jobs = sessions
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| indices.contains(i))
                    .map(|(_, s)| {
                        s.local[..h].fill(0.);
                        (&mut s.kernels, s.position, &mut s.local[..h])
                    })
                    .collect::<Vec<_>>();
                for (i, result) in indices.into_iter().zip(K::attention_rows(&mut jobs, li)) {
                    status[i] = result;
                }
                for (s, st) in sessions.iter_mut().zip(&mut status) {
                    if plan.ep2 {
                        s.reduced[..h].copy_from_slice(&s.local[..h]);
                        if s.local[..h].iter().any(|v| !v.is_finite()) {
                            *st = Err("EP2 nonfinite attention".into());
                        }
                    }
                }
                if !plan.ep2 {
                    Self::sum_rows(sessions, h, status.clone(), None)?;
                }
                for (s, st) in sessions.iter_mut().zip(&mut status) {
                    *st = st
                        .clone()
                        .and_then(|_| s.kernels.inject_attention(li, &s.reduced[..h]));
                }
            } else {
                let width = if !plan.ep2 && sessions[0].kernels.separate_shared() {
                    2 * h
                } else {
                    h
                };
                let indices = status
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.is_ok().then_some(i))
                    .collect::<Vec<_>>();
                let mut jobs = sessions
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| indices.contains(i))
                    .map(|(_, s)| {
                        s.local[..width].fill(0.);
                        (&mut s.kernels, &mut s.local[..width])
                    })
                    .collect::<Vec<_>>();
                for (i, result) in indices.into_iter().zip(K::moe_rows(&mut jobs, li)) {
                    status[i] = result;
                }
                let engine_model::shapes::Ffn::Experts(e) = plan.shape.layers[li].ffn else {
                    return Err("Next rows require MoE".into());
                };
                Self::sum_rows(
                    sessions,
                    width,
                    status.clone(),
                    Some((e.selected as usize, e.count)),
                )?;
                for (s, st) in sessions.iter_mut().zip(&mut status) {
                    if width == 2 * h {
                        for i in 0..h {
                            s.reduced[i] += s.reduced[h + i];
                        }
                    }
                    if plan.ep2 {
                        s.kernels.add_replicated_shared(&mut s.reduced[..h])?;
                    }
                    *st = s.kernels.inject_moe(li, &s.reduced[..h]);
                }
            }
            Ok(())
        })?;
        for (s, st) in sessions.iter_mut().zip(status) {
            s.prepare_head(st);
        }
        Self::gather_rows(sessions, 4)?;
        sessions.iter_mut().map(|s| s.finish_head()).collect()
    }
    /// Consecutive prompt tokens of this session as one chunk (layer-major, one gather
    /// per existing boundary for all rows); `head` = the last row's prediction is read.
    /// Every row keeps the single-token arithmetic, causal order and ascending rank fold,
    /// so the result and the causal state equal advance()/step() token by token.
    pub fn chunk(&mut self, tokens: &[u32], head: bool) -> Result<u32> {
        let n = tokens.len();
        if n == 0 {
            return Err("empty Next chunk".into());
        }
        if n == 1 || n > self.kernels.chunk_rows() || self.plan().ep2 {
            let (&last, prefix) = tokens.split_last().unwrap();
            for &t in prefix {
                self.advance(t)?;
            }
            return if head { self.step(last) } else { self.advance(last).map(|_| 0) };
        }
        let timer = crate::next_timing::start();
        let vocab = self.plan().shape.vocab;
        let valid = !self.failed
            && self.position + n <= self.capacity
            && tokens.iter().all(|&t| u64::from(t) < vocab);
        // The ordered token list folds into the agreed token word.
        let folded = tokens
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &t| (h ^ u64::from(t)).wrapping_mul(0x100_0000_01b3));
        self.agree(4 + 16 * n as u64 + 8 * u64::from(head), folded as u32 ^ (folded >> 32) as u32, valid)?;
        let result = self.chunk_inner(tokens, head);
        if result.is_err() {
            self.failed = true;
        }
        crate::next_timing::step(timer, self.transport.rank(), n);
        result
    }
    fn chunk_inner(&mut self, tokens: &[u32], head: bool) -> Result<u32> {
        let n = tokens.len();
        let h = self.plan().shape.hidden as usize;
        let mut local = vec![0f32; n * 2 * h];
        let mut status = Ok(());
        for (i, &t) in tokens.iter().enumerate() {
            let row = &mut local[i * 2 * h..(i + 1) * 2 * h];
            let s = self.kernels.front(&self.model.checkpoint, t, self.position + i, row);
            status = status.and(s);
        }
        let mut reduced = vec![0f32; n * 2 * h];
        self.sum_chunk(&local, &mut reduced, 2 * h, n, status, None)?;
        let mut status = self.kernels.embed_chunk(&reduced, n);
        let separate = self.kernels.separate_shared();
        let width = if separate { 2 * h } else { h };
        let mut local = vec![0f32; n * width];
        for layer in 0..self.plan().shape.layers.len() {
            local[..n * h].fill(0.);
            status = status.and_then(|_| self.kernels.attention_chunk(layer, &mut local[..n * h], n));
            self.sum_chunk(&local[..n * h], &mut reduced[..n * h], h, n, status.clone(), None)?;
            status = status.and_then(|_| self.kernels.inject_attention_chunk(layer, &reduced[..n * h], n));
            local.fill(0.);
            status = status.and_then(|_| self.kernels.moe_chunk(layer, &mut local, n));
            let engine_model::shapes::Ffn::Experts(experts) = self.plan().shape.layers[layer].ffn
            else {
                return Err("Next skeleton requires MoE layers".into());
            };
            self.sum_chunk(
                &local,
                &mut reduced[..n * width],
                width,
                n,
                status,
                Some((experts.selected as usize, experts.count)),
            )?;
            let mut block = vec![0f32; n * h];
            for (r, out) in block.chunks_exact_mut(h).enumerate() {
                let row = &reduced[r * width..(r + 1) * width];
                out.copy_from_slice(&row[..h]);
                if separate {
                    for i in 0..h {
                        out[i] += row[h + i];
                    }
                }
            }
            status = self.kernels.inject_moe_chunk(layer, &block, n);
        }
        status = status.and_then(|_| self.kernels.finish_chunk(n));
        self.position += n - 1;
        if head {
            self.prepare_head(status);
            let world = self.transport.world();
            self.transport.floats(&mut self.wire[..4 * world], 4)?;
            return self.finish_head();
        }
        let (world, rank) = (self.transport.world(), self.transport.rank());
        self.wire[rank * 4..(rank + 1) * 4]
            .copy_from_slice(&[if status.is_ok() { 1. } else { 0. }, 0., 0., 0.]);
        self.transport.floats(&mut self.wire[..4 * world], 4)?;
        if self.wire[..4 * world]
            .chunks_exact(4)
            .any(|peer| peer[0] != 1. || peer.iter().any(|x| !x.is_finite()))
        {
            return Err("Next rank prompt chunk failed".into());
        }
        self.position += 1;
        Ok(0)
    }
    /// sum() for `n` rows in gathers of whole rows that fit one rank's mailbox slot: each
    /// row is prepare_sum's status/values/route row and folds exactly as finish_sum
    /// (ascending rank order, per element). Every rank runs the same group schedule.
    fn sum_chunk(
        &mut self,
        local: &[f32],
        reduced: &mut [f32],
        width: usize,
        n: usize,
        status: Result<()>,
        route: Option<(usize, u64)>,
    ) -> Result<()> {
        let extra = route.map_or(0, |(top, _)| top * 2);
        let row = width + 1 + extra;
        let group = (RANK_MAILBOX_SLOT / (row * 4)).max(1);
        for first in (0..n).step_by(group) {
            let rows = group.min(n - first);
            self.sum_group(
                &local[first * width..(first + rows) * width],
                &mut reduced[first * width..(first + rows) * width],
                width,
                first,
                rows,
                status.clone(),
                route,
            )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn sum_group(
        &mut self,
        local: &[f32],
        reduced: &mut [f32],
        width: usize,
        first: usize,
        n: usize,
        status: Result<()>,
        route: Option<(usize, u64)>,
    ) -> Result<()> {
        let (rank, world) = (self.transport.rank(), self.transport.world());
        let extra = route.map_or(0, |(top, _)| top * 2);
        let row = width + 1 + extra;
        let mut wire = vec![0f32; world * n * row];
        for r in 0..n {
            let routes = route.map_or(&[][..], |_| self.kernels.chunk_routing(first + r));
            let values = &local[r * width..(r + 1) * width];
            let valid_route = route.is_none_or(|(top, count)| {
                routes.len() == top
                    && routes.iter().all(|&(id, weight)| {
                        u64::from(id) < count && weight.is_finite() && weight >= 0.
                    })
                    && routes.windows(2).all(|pair| pair[0].0 < pair[1].0)
            });
            let valid = status.is_ok() && valid_route && all_finite(values);
            let dst = &mut wire[(rank * n + r) * row..(rank * n + r + 1) * row];
            dst[0] = if valid { 1. } else { 0. };
            dst[1..1 + width].copy_from_slice(values);
            if valid {
                for (d, &(id, weight)) in dst[1 + width..].chunks_exact_mut(2).zip(routes) {
                    d.copy_from_slice(&[id as f32, weight]);
                }
            }
        }
        self.transport.floats(&mut wire, n * row)?;
        for r in 0..n {
            let peer = |p: usize| &wire[(p * n + r) * row..(p * n + r + 1) * row];
            if (0..world).any(|p| peer(p)[0] != 1. || !all_finite(&peer(p)[1..])) {
                return Err(status
                    .err()
                    .unwrap_or_else(|| "Next rank kernel failed or produced nonfinite output".into()));
            }
            if (0..world).any(|p| {
                peer(p)[1 + width..]
                    .iter()
                    .zip(&peer(0)[1 + width..])
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            }) {
                return Err("Next rank routing mismatch".into());
            }
            let out = &mut reduced[r * width..(r + 1) * width];
            let finite = sum_rows((0..world).map(|p| &peer(p)[1..1 + width]), out);
            if !finite {
                return Err("Next reduction overflow".into());
            }
        }
        Ok(())
    }
    /// Chunk boundaries and interleaved sessions never change the token loop.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        tokens.iter().map(|&t| self.step(t)).collect()
    }
    pub fn reset(&mut self) -> Result<()> {
        self.agree(2, 0, true)?;
        self.local.fill(0.);
        let status = self.kernels.reset();
        self.sum(0, status, None)
            .inspect_err(|_| self.failed = true)?;
        self.position = 0;
        self.failed = false;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or("Next reset epoch overflow")?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn sum_pair_matches_copy_add_and_overflow_check() {
        let mut seed = 0x9e37_79b9u32;
        let mut bits = vec![0, 0x8000_0000, 1, 0x007f_ffff, 0x0080_0000,
            0x3f80_0000, 0xbf80_0000, 0x7f7f_ffff, 0xff7f_ffff];
        for _ in 0..4096 {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            if f32::from_bits(seed).is_finite() { bits.push(seed); }
        }
        for width in [0, 1, 3, 16, 63, 128, 2560, 5120, 56320] {
            let left = (0..width).map(|i| f32::from_bits(bits[i % bits.len()])).collect::<Vec<_>>();
            let right = (0..width).map(|i| f32::from_bits(bits[(i * 19 + 3) % bits.len()])).collect::<Vec<_>>();
            let mut expected = left.clone();
            for (a, b) in expected.iter_mut().zip(&right) { *a += b; }
            let mut actual = vec![0f32; width];
            assert_eq!(super::sum_pair(&left, &right, &mut actual), super::all_finite(&expected));
            assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "width {width}");
        }
        assert!(!super::sum_pair(&[f32::MAX], &[f32::MAX], &mut [0.]));
    }
    #[test]
    fn ordered_row_fold_matches_rank_order_including_intermediate_overflow() {
        for count in 1..=8 {
            for width in [0, 1, 3, 128, 2560] {
                let rows = (0..count).map(|rank| (0..width).map(|i| {
                    [0., -0., f32::MAX, -f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1),
                        (i as f32 - 128.) * (rank as f32 - 2.)][(i + rank) % 7]
                }).collect::<Vec<_>>()).collect::<Vec<_>>();
                let mut expected = rows[0].clone();
                for row in &rows[1..] { for (v, a) in expected.iter_mut().zip(row) { *v += a; } }
                let mut actual = vec![0.; width];
                assert_eq!(super::sum_rows(rows.iter().map(Vec::as_slice), &mut actual),
                    super::all_finite(&expected));
                assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
        }
        let rows = [[f32::MAX], [f32::MAX], [-f32::MAX]];
        assert!(!super::sum_rows(rows.iter().map(|v| v.as_slice()), &mut [0.]));
    }
    #[test]
    fn all_finite_matches_is_finite() {
        let special = [
            0., -0., 1., -1., f32::MIN, f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1),
            f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN, f32::from_bits(0x7f80_0001),
        ];
        for &a in &special {
            for &b in &special {
                let v = [1.5, a, -2., b];
                assert_eq!(super::all_finite(&v), v.iter().all(|x| x.is_finite()), "{a} {b}");
            }
        }
        assert!(super::all_finite(&[]));
    }
}
