//! Session arenas (VADE_SESSION_ARENA): a Flash-Next session's GDN states and its MTP
//! rollback storage carved from one allocation each. Split from gdn.rs (file-size ratchet).
use super::{chunk_form, Backend, DeviceVector, GdnReplay, GdnState, Result, Weights, DIM};

/// VADE_SESSION_ARENA (default on; =0 one allocation per vector, as before): a Flash-Next
/// session's GDN states, and its MTP rollback storage (entry snapshots + replay tapes), are
/// views of one allocation each instead of 36 and 108 separate cuMemAlloc/cuMemFree pairs
/// (nsys: ~0.2 ms per state allocation, ~0.07 ms per free, GPU idle meanwhile). Same lengths,
/// same contents; vectors whose byte length is a multiple of 4096 come first, so each of them
/// stays 4 KiB-aligned (the rest keep vector_arena's 64-byte alignment, as GdnReplay's tails do).
pub(crate) fn session_arena() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VADE_SESSION_ARENA").is_ok_and(|v| v == "0"))
}
/// One allocation for `lengths` (FP32 elements), returned in the given order.
fn ordered_arena<'a>(b: &Backend<'a>, lengths: &[usize]) -> Result<Vec<DeviceVector<'a>>> {
    if lengths.is_empty() { return Ok(Vec::new()); }
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| !(lengths[i] * 4).is_multiple_of(4096));
    let sorted: Vec<usize> = order.iter().map(|&i| lengths[i]).collect();
    let mut slots: Vec<Option<DeviceVector<'a>>> = (0..lengths.len()).map(|_| None).collect();
    for (i, v) in order.into_iter().zip(b.vector_arena(&sorted)?) { slots[i] = Some(v); }
    slots.into_iter().map(|v| v.ok_or_else(|| "session arena shape".to_string())).collect()
}

impl<'a> GdnState<'a> {
    /// Prefix cache: FP32 lengths of the state and history storage (one arena per checkpoint).
    pub(in crate::flash) fn storage_lengths(&self) -> [usize; 2] { [self.state.len(), self.history.len()] }
    /// Storage for this layer in caller-allocated vectors of `storage_lengths`.
    pub(in crate::flash) fn with_storage(&self, state: DeviceVector<'a>, history: DeviceVector<'a>) -> Self {
        Self { layer: self.layer, rank: self.rank, state, history, cached: self.cached, half: self.half }
    }
    /// Fresh (zeroed, uncached) states for `layers`, sized as Gdn::load_next sizes a fresh
    /// state, carved from one allocation (session_arena).
    pub(in crate::flash) fn fresh_arena(b: &Backend<'a>, w: &Weights<'_>, layers: &[usize]) -> Result<Vec<Self>> {
        let heads = (w.plan.value_heads.end - w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end - w.plan.key_heads.start) as usize;
        let width = (2 * keys + heads) * DIM;
        let recurrent = if chunk_form() && heads != 0 {
            spark_gpu_probe::gdn_params::chunk_state_floats(heads, keys)
        } else { heads * DIM * DIM };
        let lengths: Vec<usize> = layers.iter().flat_map(|_| [recurrent.max(1), (width * 4).max(1)]).collect();
        let mut v = ordered_arena(b, &lengths)?.into_iter();
        layers.iter().map(|&layer| {
            let (state, history) = (v.next().ok_or("GDN arena shape")?, v.next().ok_or("GDN arena shape")?);
            let mut s = Self { layer, rank: w.plan.rank, state, history, cached: false, half: false };
            s.reset(b)?;
            Ok(s)
        }).collect()
    }
    /// Entry-snapshot storage for every live state (as next_snapshot_storage) and a replay
    /// tape of `rows` rows for every `tapes` entry that is true (as GdnReplay::new), carved
    /// from one allocation (session_arena).
    #[allow(clippy::type_complexity)]
    pub(in crate::flash) fn rollback_arena(b: &Backend<'a>, live: &[Option<Self>], tapes: &[bool], rows: usize)
        -> Result<(Vec<Option<Self>>, Vec<Option<GdnReplay<'a>>>)> {
        let mut lengths = Vec::new();
        for s in live.iter().flatten() { lengths.extend([s.state.len(), s.history.len()]); }
        for _ in tapes.iter().filter(|&&t| t) { lengths.extend(GdnReplay::widths(rows)); }
        let mut v = ordered_arena(b, &lengths)?.into_iter();
        let mut next = || v.next().ok_or_else(|| "GDN rollback arena shape".to_string());
        let gdn = live.iter().map(|s| match s {
            Some(s) => Ok(Some(Self { layer: s.layer, rank: s.rank, state: next()?, history: next()?, cached: s.cached, half: s.half })),
            None => Ok(None),
        }).collect::<Result<Vec<_>>>()?;
        let inputs = tapes.iter().map(|&t| if t {
            Ok(Some(GdnReplay::from_projected([next()?, next()?, next()?])))
        } else { Ok(None) }).collect::<Result<Vec<_>>>()?;
        Ok((gdn, inputs))
    }
}
