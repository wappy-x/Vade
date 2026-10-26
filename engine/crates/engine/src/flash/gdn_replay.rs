//! Exact rollback tapes for world-one Flash-Next. Store projections, never
//! rounded deltas or inverse updates: replay preserves the ascending FP32 fold.
use super::*;

const WIDTHS: [usize; 3] = [10240, 48, 48];

pub(in crate::flash) struct GdnReplay<'a> {
    projected: [DeviceVector<'a>; 3],
    rows: usize,
    /// Per-row words [qkv, a, b] of the captured rows: WIDTHS, or a VADE_W2_TP_PROJ=3 head half's
    /// (Gdn::tape_widths), which fits the same storage.
    widths: [usize; 3],
}

impl<'a> GdnReplay<'a> {
    pub(in crate::flash) fn reservation_bytes(rows: usize) -> usize {
        WIDTHS
            .into_iter()
            .map(|n| (n * rows * 4).div_ceil(64) * 64)
            .sum()
    }

    /// Tape lengths (FP32 elements) for `rows` rows, in `projected` order.
    pub(in crate::flash) fn widths(rows: usize) -> [usize; 3] {
        WIDTHS.map(|n| n * rows)
    }

    /// A tape over caller-provided storage of `widths(rows)` lengths (gdn::session_arena).
    pub(in crate::flash) fn from_projected(projected: [DeviceVector<'a>; 3]) -> Self {
        Self { projected, rows: 0, widths: WIDTHS }
    }

    pub(in crate::flash) fn new(b: &Backend<'a>, rows: usize) -> Result<Self> {
        let projected = b
            .vector_arena(&WIDTHS.map(|n| n * rows))?
            .try_into()
            .map_err(|_| "GDN replay arena shape")?;
        Ok(Self { projected, rows: 0, widths: WIDTHS })
    }

    /// capture's three copies as (source, destination, words) for flash_gdn_verify_copy
    /// (gdn::fused_verify_copy); the launch must run before the layer scratch is reused.
    pub(in crate::flash) fn capture_ranges(
        &mut self,
        s: &GdnScratch<'a>,
        start: usize,
        rows: usize,
        widths: [usize; 3],
    ) -> Result<[(u64, u64, u32); 3]> {
        self.rows = 0;
        self.widths = widths;
        let mut out = [(0u64, 0u64, 0u32); 3];
        for (((source, target), width), slot) in [&s.qkv, &s.a, &s.b]
            .into_iter()
            .zip(&self.projected)
            .zip(widths)
            .zip(out.iter_mut())
        {
            let (offset, len) = (start * width, rows * width);
            if offset.checked_add(len).is_none_or(|n| n > source.len()) || len > target.len()
                || source.is_segmented() || target.is_segmented() {
                return Err("GDN replay tape copy exceeds vector".into());
            }
            *slot = (source.device_address() + (offset * 4) as u64, target.device_address(),
                u32::try_from(len).map_err(|_| "GDN replay tape words")?);
        }
        self.rows = rows;
        Ok(out)
    }

    /// capture's three copies as flash_hybrid_copy rows [qkv, a, b] (gdn::cohort_copies); the
    /// caller launches them before the layer scratch is reused.
    pub(in crate::flash) fn capture_rows(
        &mut self,
        s: &GdnScratch<'a>,
        start: usize,
        rows: usize,
    ) -> Result<[spark_gpu_probe::kernel_params::HybridCopyRow; 3]> {
        self.rows = 0;
        self.widths = WIDTHS;
        let mut out = [spark_gpu_probe::kernel_params::HybridCopyRow { source: 0, destination: 0, words: 0, padding: 0 }; 3];
        for (((source, target), width), row) in [&s.qkv, &s.a, &s.b].into_iter().zip(&self.projected).zip(WIDTHS).zip(&mut out) {
            let words = rows * width;
            if (start + rows) * width > source.len() || words > target.len() || source.is_segmented() || target.is_segmented() {
                return Err("GDN replay capture outside its vectors".into());
            }
            *row = spark_gpu_probe::kernel_params::HybridCopyRow { source: source.device_address() + (start * width * 4) as u64,
                destination: target.device_address(), words: u32::try_from(words).map_err(|_| "GDN replay words")?, padding: 0 };
        }
        self.rows = rows;
        Ok(out)
    }
    /// Copy just QKV and gate projections before the layer scratch is reused.
    pub(in crate::flash) fn capture(
        &mut self,
        b: &Backend<'a>,
        s: &GdnScratch<'a>,
        start: usize,
        rows: usize,
        widths: [usize; 3],
    ) -> Result<()> {
        self.rows = 0;
        self.widths = widths;
        for ((source, target), width) in [&s.qkv, &s.a, &s.b]
            .into_iter()
            .zip(&mut self.projected)
            .zip(widths)
        {
            b.copy_range(source, target, start * width, 0, rows * width)?;
        }
        self.rows = rows;
        Ok(())
    }
}

impl<'a> Gdn<'a, '_> {
    /// Shallow rejection (gdn::shallow_verify): S is untouched and the
    /// accepted rows' tail entries are in place (the verify wrote the same
    /// bytes the row replay would), so only the conv history advances over
    /// the accepted rows: one history launch over all of them, the same
    /// copied words as one launch per row.
    pub(in crate::flash) fn replay_history(&mut self, b: &Backend<'a>, tape: &GdnReplay<'a>, rows: usize) -> Result<()> {
        use spark_gpu_probe::kernel_params::GdnHistory;
        if rows > tape.rows || tape.widths != self.tape_widths() {
            return Err("GDN replay prefix outside retained projections".into());
        }
        let width = (2 * self.keys + self.heads) * DIM;
        let mut history = GdnHistory {
            x: tape.projected[0].device_address() as _,
            history: self.history.device_address() as _,
            rows: 0,
            channels: width as u32,
            tokens: rows as u32,
            precision: 0,
        };
        unsafe {
            b.launch_flash_shared("flash_gdn_history", width.div_ceil(128) as u32, 128, 0,
                &mut [(&mut history as *mut GdnHistory).cast()])?;
        }
        self.cached = true;
        Ok(())
    }
    /// Reuse each session's funded one-row scratch. Rejection needs no dense,
    /// attention, expert or head work, and leaves the original scratch views intact.
    pub(in crate::flash) fn replay_next(
        &mut self,
        b: &Backend<'a>,
        s: &mut GdnScratch<'a>,
        tape: &GdnReplay<'a>,
        rows: usize,
        descriptor: &mut DeviceVector<'a>,
        base: usize,
    ) -> Result<()> {
        if rows > tape.rows || tape.widths != self.tape_widths() {
            return Err("GDN replay prefix outside retained projections".into());
        }
        for row in 0..rows {
            let [qkv, a, gate_b] = std::array::from_fn(|i| {
                tape.projected[i].slice(row * tape.widths[i]..(row + 1) * tape.widths[i])
            });
            let (qkv, a, gate_b) = (qkv?, a?, gate_b?);
            let original = [
                std::mem::replace(&mut s.qkv, qkv),
                std::mem::replace(&mut s.a, a),
                std::mem::replace(&mut s.b, gate_b),
            ];
            let result = self.scan_next(b, s, (0, 1), descriptor, None, false, base + row, None, None);
            [s.qkv, s.a, s.b] = original;
            result?;
        }
        Ok(())
    }
}
