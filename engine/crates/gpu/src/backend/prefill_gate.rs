//! Safe host launch wrappers for fused prefill message-meta and message-gate kernels
//! (`PREFILL_GATE_FUSE`, kind 123).
use super::*;
use crate::prefill_gate::{
    self, GateSpans, PG_FLAG_HALF_OUT, PG_FULL_THREADS, PG_GATE_AT, PG_GATE_STRIDE,
    PG_META_THREADS, PG_OP_FULL, PG_OP_MSG_META, PG_OP_WORD, Span as PgSpan, full_admitted,
    full_blocks, msg_meta_admitted, msg_meta_blocks,
};
use crate::resident_launch::{Command, Kind};

/// One gate entry passed to `Backend::prefill_gate_fuse`.
pub type PrefillGateEntry<'a, 'd> = (
    &'a DeviceVector<'d>,         // w_down [k, 2 * dim] BF16
    &'a DeviceVector<'d>,         // w_outer [2 * dim, k] BF16 ([GateForget, GateGain])
    &'a DeviceVector<'d>,         // kept_val [rows, dim] FP32
    &'a DeviceVector<'d>,         // kept_normed_half [rows, dim / 2] packed BF16
    &'a DeviceVector<'d>,         // msg_weights [rows, kept] FP32
    &'a DeviceVector<'d>,         // msg_nscales [rows] FP32
    Option<&'a DeviceVector<'d>>, // optional norm_w [dim] FP32
);

impl<'a> Backend<'a> {
    /// True when `PG_OP_MSG_META` can execute this shape and operand alignment.
    pub fn prefill_msg_meta_ready(
        &self, rows: usize, asked: usize, dim: usize,
        queries: &DeviceVector<'_>, sources: &[&DeviceVector<'_>],
    ) -> bool {
        if !self.resident_launches_active()
            || !msg_meta_admitted(rows, asked, sources.len(), dim, PG_META_THREADS)
            || queries.is_segmented() || (queries.device_address() & 15) != 0
        {
            return false;
        }
        sources.iter().all(|s| !s.is_segmented() && (s.device_address() & 15) == 0 && s.len() == rows * dim)
    }

    /// Launch `PG_OP_MSG_META` (kind 123, sub-op 0): computes compact lazy `(msg_weights, msg_nscales)`
    /// for `Token::enter`, replacing `MIX_LIST_NORM_HALF`'s 2.72 GB/pass global materialization.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_msg_meta(
        &self, rows: usize, dim: usize, scale: f32, eps: f32,
        queries: &DeviceVector<'_>, sources: &[&DeviceVector<'_>],
    ) -> Result<(DeviceVector<'a>, DeviceVector<'a>), String> {
        if self.arm_capture() { return Err("prefill msg meta: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        if dim == 0 || !queries.len().is_multiple_of(dim) {
            return Err("prefill msg meta: queries length not multiple of dim".into());
        }
        let asked = queries.len() / dim;
        let kept = sources.len();
        self.check_vector(queries, asked * dim)?;
        if queries.is_segmented() { return Err("prefill msg meta: queries segmented".into()); }
        for src in sources {
            self.check_vector(src, rows * dim)?;
            if src.is_segmented() { return Err("prefill msg meta: source segmented".into()); }
        }
        let mut weights_out = self.vector(asked * rows * kept)?;
        let mut nscales_out = self.vector(asked * rows)?;
        if weights_out.is_segmented() || !weights_out.writable_from(0)
            || nscales_out.is_segmented() || !nscales_out.writable_from(0)
        {
            return Err("prefill msg meta: outputs require private contiguous storage".into());
        }
        let span = |v: &DeviceVector<'_>| PgSpan { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let src_spans: Vec<PgSpan> = sources.iter().map(|s| span(s)).collect();
        prefill_gate::check_msg_meta(rows, asked, dim, scale, eps, span(queries), &src_spans, span(&weights_out), span(&nscales_out))?;
        self.require_arm_kernels(&[Kind::PrefillGateFuse.kernel()])?;
        self.skip_fill(&mut weights_out)?;
        self.skip_fill(&mut nscales_out)?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::PrefillGateFuse);
        command.word(PG_OP_WORD, PG_OP_MSG_META);
        command.word(1, queries.device_address());
        command.word(2, weights_out.device_address());
        command.word(3, nscales_out.device_address());
        command.word(4, rows as u64);
        command.word(5, asked as u64);
        command.word(6, kept as u64);
        command.word(7, dim as u64);
        command.word(8, u64::from(scale.to_bits()) | (u64::from(eps.to_bits()) << 32));
        for (i, src) in sources.iter().enumerate() {
            command.word(9 + i, src.device_address());
        }
        let grid = [msg_meta_blocks(rows, kept), 1, 1, PG_META_THREADS as u32, 0];
        if !self.record_resident_block(Kind::PrefillGateFuse, command, grid)? {
            return Err("prefill msg meta: launches are not recorded now".into());
        }
        self.resident_completion()?;
        Ok((weights_out, nscales_out))
    }

    /// Launch `PG_OP_FULL` (kind 123, sub-op 1): fused `GateDown + GateOuter + GateBlend + RoundedNorm`
    /// for `1..=3` gates directly from lazy sources `sources[0..kept]`, returning `count` outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_gate_fuse(
        &self, rows: usize, dim: usize, k: usize, eps: f32, half_out: bool,
        sources: &[&DeviceVector<'_>], gates: &[PrefillGateEntry<'_, '_>],
    ) -> Result<Vec<DeviceVector<'a>>, String> {
        if self.arm_capture() { return Err("prefill gate fuse: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let (count, kept) = (gates.len(), sources.len());
        if !full_admitted(rows, dim, k, count, kept, PG_FULL_THREADS) {
            return Err("prefill gate fuse: shape not admitted".into());
        }
        for src in sources {
            self.check_vector(src, rows * dim)?;
            if src.is_segmented() { return Err("prefill gate fuse: source segmented".into()); }
        }
        let out_words = if half_out { rows * (dim / 2) } else { rows * dim };
        let mut outs = Vec::with_capacity(count);
        for &(wd, wo, kv, kh, mw, mn, nw) in gates {
            self.check_vector(wd, k * dim)?;
            self.check_vector(wo, dim * k)?;
            self.check_vector(kv, rows * dim)?;
            self.check_vector(kh, rows * (dim / 2))?;
            self.check_vector(mw, rows * kept)?;
            self.check_vector(mn, rows)?;
            if wd.is_segmented() || wo.is_segmented() || kv.is_segmented()
                || kh.is_segmented() || mw.is_segmented() || mn.is_segmented()
            {
                return Err("prefill gate fuse: gate input segmented".into());
            }
            if let Some(n) = nw {
                self.check_vector(n, dim)?;
                if n.is_segmented() { return Err("prefill gate fuse: norm_w segmented".into()); }
            }
            let out = self.vector(out_words)?;
            if out.is_segmented() || !out.writable_from(0) {
                return Err("prefill gate fuse: output requires private contiguous storage".into());
            }
            outs.push(out);
        }
        let span = |v: &DeviceVector<'_>| PgSpan { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let src_spans: Vec<PgSpan> = sources.iter().map(|s| span(s)).collect();
        let gate_spans: Vec<GateSpans> = gates.iter().zip(&outs).map(|(&(wd, wo, kv, kh, mw, mn, nw), o)| GateSpans {
            w_down: span(wd), w_outer: span(wo), kept_val: span(kv), kept_normed_half: span(kh),
            msg_weights: span(mw), msg_nscales: span(mn), norm_w: nw.map(span), output: span(o),
        }).collect();
        prefill_gate::check_full(rows, dim, k, eps, half_out, &src_spans, &gate_spans)?;
        self.require_arm_kernels(&[Kind::PrefillGateFuse.kernel()])?;
        for o in &mut outs { self.skip_fill(o)?; }
        self.resident_completion()?;
        let mut command = Command::new(Kind::PrefillGateFuse);
        command.word(PG_OP_WORD, PG_OP_FULL);
        command.word(1, rows as u64);
        command.word(2, dim as u64);
        command.word(3, k as u64);
        command.word(4, count as u64);
        command.word(5, kept as u64);
        let flags = if half_out { PG_FLAG_HALF_OUT } else { 0 };
        command.word(6, u64::from(eps.to_bits()) | (flags << 32));
        for (i, src) in sources.iter().enumerate() {
            command.word(7 + i, src.device_address());
        }
        for (gi, (&(wd, wo, kv, kh, mw, mn, nw), o)) in gates.iter().zip(&outs).enumerate() {
            let base = PG_GATE_AT + gi * PG_GATE_STRIDE;
            command.word(base, wd.device_address());
            command.word(base + 1, wo.device_address());
            command.word(base + 2, kv.device_address());
            command.word(base + 3, kh.device_address());
            command.word(base + 4, mw.device_address());
            command.word(base + 5, mn.device_address());
            command.word(base + 6, nw.map_or(0, DeviceVector::device_address));
            command.word(base + 7, o.device_address());
        }
        let grid = [full_blocks(rows, count), 1, 1, PG_FULL_THREADS as u32, 0];
        if !self.record_resident_block(Kind::PrefillGateFuse, command, grid)? {
            return Err("prefill gate fuse: launches are not recorded now".into());
        }
        self.resident_completion()?;
        Ok(outs)
    }
}
