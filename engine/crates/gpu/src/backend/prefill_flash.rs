//! Safe host launch wrapper for high-occupancy 2-stage prefill attention
//! (`PREFILL_FLASH_WINDOW`, kind 122).
use super::*;
use crate::flash_window::{
    self, FW_CLIP_IN, FW_CLIP_WORD, FW_HALF_OUT, Span as FwSpan,
    flash_window_prefill_admitted, flash_window_prefill_blocks, flash_window_prefill_threads,
};
use crate::resident_launch::{Command, Kind};

impl<'a> Backend<'a> {
    /// True when the high-occupancy 2-stage `PREFILL_FLASH_WINDOW` kernel (kind 122) can execute this shape/alignment.
    #[allow(clippy::too_many_arguments, reason = "shape and device operand alignments")]
    pub fn prefill_flash_window_ready(
        &self, rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize,
        query: &DeviceVector<'_>, keys: &DeviceVector<'_>, values: &DeviceVector<'_>,
        moved: &DeviceVector<'_>, rel_table: &DeviceVector<'_>, gate: &DeviceVector<'_>,
        output: &DeviceVector<'_>,
    ) -> bool {
        let threads = flash_window_prefill_threads(heads, kv_heads);
        flash_window_prefill_admitted(
            rows, heads, kv_heads, view, extent,
            query.device_address() as usize, keys.device_address() as usize,
            values.device_address() as usize, moved.device_address() as usize,
            rel_table.device_address() as usize, gate.device_address() as usize,
            output.device_address() as usize, threads,
        )
    }

    /// Launch high-occupancy 2-stage prefill `PREFILL_FLASH_WINDOW` (kind 122), bit-exact with `FLASH_WINDOW_ATTN`.
    #[allow(clippy::too_many_arguments, reason = "shape, scale, output form and nine device operands")]
    pub fn prefill_flash_window_into(
        &self, rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize,
        scale: f32, first_position: u64, half_out: bool, query: &DeviceVector<'_>, keys: &DeviceVector<'_>,
        values: &DeviceVector<'_>, moved: &DeviceVector<'_>, rel_table: &DeviceVector<'_>, sink: &DeviceVector<'_>,
        gate: &DeviceVector<'_>, steps: &DeviceVector<'_>, clips: Option<OneClips<'_, '_>>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.prefill_flash_window_form_into(
            rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, first_position, half_out, false,
            query, keys, values, moved, rel_table, sink, gate, steps, clips, output,
        )
    }

    /// `prefill_flash_window_into` with the form of `query` and `gate`: FP32 words, or with `qg_half` the BF16 pairs
    /// of `flash_window::FW_QG_HALF` (`rows * heads * head_dim / 2` words each; the launch needs `clips`).
    #[allow(clippy::too_many_arguments, reason = "shape, scale, operand forms and nine device operands")]
    pub fn prefill_flash_window_form_into(
        &self, rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize,
        scale: f32, first_position: u64, half_out: bool, qg_half: bool, query: &DeviceVector<'_>, keys: &DeviceVector<'_>,
        values: &DeviceVector<'_>, moved: &DeviceVector<'_>, rel_table: &DeviceVector<'_>, sink: &DeviceVector<'_>,
        gate: &DeviceVector<'_>, steps: &DeviceVector<'_>, clips: Option<OneClips<'_, '_>>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("prefill flash window: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        if qg_half && clips.is_none() { return Err("prefill flash window: the half query form needs the clips".into()); }
        let span = |v: &DeviceVector<'_>| FwSpan { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let kv_words = view * kv_heads * (head_dim / 2);
        let prefix_span = |v: &DeviceVector<'_>, words: usize| FwSpan { address: v.device_address(), bytes: words as u64 * 4 };
        flash_window::check_flash_window_form_in(
            rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, half_out, qg_half,
            span(query), prefix_span(keys, kv_words), prefix_span(values, kv_words), span(moved), span(rel_table), span(sink), span(gate), span(steps), span(output),
        )?;
        let threads = flash_window_prefill_threads(heads, kv_heads);
        if !flash_window_prefill_admitted(
            rows, heads, kv_heads, view, extent,
            query.device_address() as usize, keys.device_address() as usize,
            values.device_address() as usize, moved.device_address() as usize,
            rel_table.device_address() as usize, gate.device_address() as usize,
            output.device_address() as usize, threads,
        ) {
            return Err("prefill flash window: shape or alignment outside prefill envelope".into());
        }
        let out_words = if half_out { rows * heads * head_dim / 2 } else { rows * heads * head_dim };
        let qg_words = if qg_half { rows * heads * head_dim / 2 } else { rows * heads * head_dim };
        self.check_vector(query, qg_words)?;
        self.check_vector_prefix(keys, kv_words)?;
        self.check_vector_prefix(values, kv_words)?;
        self.check_vector(moved, rows * heads * rel_dim)?;
        self.check_vector(rel_table, rel_dim * extent)?;
        self.check_vector(sink, rows * heads)?;
        self.check_vector(gate, qg_words)?;
        self.check_vector(steps, rows * 2)?;
        self.check_vector(output, out_words)?;
        for inp in [query, moved, rel_table, sink, gate, steps] {
            if inp.is_segmented() { return Err("prefill flash window: input must be contiguous".into()); }
        }
        if output.is_segmented() || !output.writable_from(0) {
            return Err("prefill flash window: output requires private contiguous storage".into());
        }
        let meets = |a: FwSpan, b: FwSpan| a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes);
        if meets(span(output), span(keys)) || meets(span(output), span(values)) {
            return Err("prefill flash window: output overlaps KV plane".into());
        }
        if let Some(clips) = &clips {
            if clips.written_keys.is_some() || clips.written_values.is_some() {
                return Err("prefill flash window: written KV operands not supported".into());
            }
            if !(clips.query_bound > 0.0 && clips.query_bound.is_finite()
                && clips.moved_bound > 0.0 && clips.moved_bound.is_finite())
            {
                return Err("prefill flash window: clip bounds must be positive finite".into());
            }
            if let Some(qs) = clips.query_scale {
                self.check_vector(qs, heads * head_dim)?;
                if qs.is_segmented() { return Err("prefill flash window: query scale segmented".into()); }
            }
            if let Some(ms) = clips.moved_scale {
                self.check_vector(ms, heads * rel_dim)?;
                if ms.is_segmented() { return Err("prefill flash window: moved scale segmented".into()); }
            }
            if let Some(pos) = clips.position {
                self.check_vector(pos, rows)?;
                if pos.is_segmented() { return Err("prefill flash window: position factor segmented".into()); }
            }
        }
        self.require_arm_kernels(&[Kind::PrefillFlashWindow.kernel()])?;
        self.skip_fill(output)?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::PrefillFlashWindow);
        let payload = [
            query.device_address(), keys.device_address(), values.device_address(), moved.device_address(),
            rel_table.device_address(), sink.device_address(), gate.device_address(), steps.device_address(),
            output.device_address(), rows as u64, heads as u64, kv_heads as u64,
            view as u64, extent as u64, u64::from(scale.to_bits()), first_position % flash_window::FW_BKV as u64,
        ];
        for (i, word) in payload.into_iter().enumerate() { command.word(i, word); }
        let mut clip_flags = if half_out { FW_HALF_OUT } else { 0 } | if qg_half { flash_window::FW_QG_HALF } else { 0 };
        if let Some(clips) = clips {
            clip_flags |= FW_CLIP_IN;
            command.word(17, clips.query_scale.map_or(0, DeviceVector::device_address));
            command.word(18, clips.moved_scale.map_or(0, DeviceVector::device_address));
            command.word(19, clips.position.map_or(0, DeviceVector::device_address));
            command.word(20, u64::from(clips.query_bound.to_bits()));
            command.word(21, u64::from(clips.moved_bound.to_bits()));
        }
        command.word(FW_CLIP_WORD, clip_flags);
        let blocks = flash_window_prefill_blocks(rows, heads, kv_heads);
        let grid = [blocks, 1, 1, threads as u32, 0];
        if !self.record_resident_block(Kind::PrefillFlashWindow, command, grid)? {
            return Err("prefill flash window: launches are not recorded now".into());
        }
        self.resident_completion()
    }
}
