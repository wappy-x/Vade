//! The one-launch forms of a visit for `Device`: the attention of one query row (`Compute::window_one`) and the key
//! or value row of a token (`Compute::conv_finish`). Each gives `Ok(None)` outside its checked form, with nothing
//! recorded, and the caller runs the separate arms.
use super::*;

impl<'b, 'd> Device<'b, 'd> {
    pub(super) fn record_window_one(&self, a: &WindowOne<'_, Value<'b, 'd>>, planes: &[(Value<'b, 'd>, Value<'b, 'd>)]) -> Result<Option<Value<'b, 'd>>> {
        use spark_gpu_probe::flash_window;
        let (rows,heads,kv_heads,view,extent)=(a.rows,a.heads,a.kv_heads,a.view,a.extent);
        if !self.backend.resident_launches_active() || !flash_window::flash_window_one_admitted(rows, heads, kv_heads, view, extent,
            a.keys.vector.device_address() as usize, a.values.vector.device_address() as usize, flash_window::FW_THREADS, rows * heads * flash_window::flash_window_one_split(rows), flash_window::flash_window_one_split(rows)) {
            return Ok(None);
        }
        let (head_dim, rel_dim) = (flash_window::FW_HEAD_DIM, flash_window::FW_REL_DIM);
        if a.query.words != rows * heads * head_dim || a.gate.words != rows * heads * head_dim || a.keys.words != view * kv_heads * (head_dim / 2)
            || a.values.words != a.keys.words || a.moved.words != rows * heads * rel_dim || a.sink.words != rows * heads || a.steps.words != 2 * rows
            || a.rel_table.len() != rel_dim * extent || !(a.query_scale.is_empty() || a.query_scale.len() == heads * head_dim)
            || !(a.moved_scale.is_empty() || a.moved_scale.len() == heads * rel_dim) || a.position.is_some_and(|p| p.words != rows)
            || !(a.query_bound > 0.0 && a.moved_bound > 0.0) {
            return Ok(None);
        }
        let written_words = rows * kv_heads * (head_dim / 2);
        if a.written.is_some() && !planes.is_empty() { return Ok(None); }
        if a.written.is_some_and(|(wk, wv)| wk.words != written_words || wv.words != written_words || wk.vector.is_segmented() || wv.vector.is_segmented()
            || wk.vector.len() != wk.words || wv.vector.len() != wv.words || (wk.vector.device_address() | wv.vector.device_address()) & 15 != 0) {
            return Ok(None);
        }
        let operands = [a.query, a.moved, a.sink, a.gate, a.steps];
        for v in operands.iter().chain(a.position.iter()) {
            if v.vector.is_segmented() || v.vector.len() != v.words { return Ok(None); }
        }
        if a.keys.vector.len() < a.keys.words || a.values.vector.len() < a.values.words {
            return Ok(None);
        }
        if (a.query.vector.device_address() | a.moved.vector.device_address() | a.gate.vector.device_address()) & 15 != 0 {
            return Ok(None);
        }
        for v in operands.iter().chain([a.keys, a.values].iter()).chain(a.position.iter()) { self.ready(v)?; }
        if let Some((wk, wv)) = a.written { self.ready(wk)?; self.ready(wv)?; }
        for (key, value) in planes {
            for plane in [key, value] {
                if plane.words != a.keys.words || plane.vector.len() < plane.words || plane.vector.device_address() % 16 != 0 { return Ok(None); }
                self.ready(plane)?;
            }
        }
        let planes = planes.iter().map(|(key,value)| (&*key.vector,&*value.vector)).collect::<Vec<_>>();
        let stamp = self.cost.start(self.backend);
        let upload = |words: &[f32]| self.parameters.borrow_mut().resolve(words, None, |values| self.backend.upload_vector(values)?.into_resident_shared());
        let table = upload(a.rel_table)?.ok_or("flash window one: relative table empty")?;
        let (query_scale, moved_scale) = (upload(a.query_scale)?, upload(a.moved_scale)?);
        let clips = spark_gpu_probe::driver::OneClips {
            query_scale: query_scale.as_deref(), moved_scale: moved_scale.as_deref(), position: a.position.map(|p| &*p.vector),
            query_bound: a.query_bound, moved_bound: a.moved_bound,
            written_keys: a.written.map(|(wk, _)| &*wk.vector), written_values: a.written.map(|(_, wv)| &*wv.vector),
        };
        let mut output = self.backend.vector(rows * heads * head_dim)?;
        self.backend.flash_window_planes_into(rows, heads, kv_heads, view, extent, a.scale, &a.query.vector, &a.keys.vector, &a.values.vector,
            &a.moved.vector, &table, &a.sink.vector, &a.gate.vector, &a.steps.vector, clips, &planes, &mut output)?;
        let value = self.value(output, rows * heads * head_dim)?;
        self.cost.record(|| "window_one".into(), stamp, self.backend, (a.rel_table.len() + a.query_scale.len() + a.moved_scale.len()) * 4);
        Ok(Some(value))
    }

    pub(super) fn record_window_split(&self, s: &WindowSplit<'_, Value<'b, 'd>>) -> Result<Option<Value<'b, 'd>>> {
        use spark_gpu_probe::flash_window;
        let a = &s.base;
        let (rows, heads, kv_heads, view, extent) = (a.rows, a.heads, a.kv_heads, a.view, a.extent);
        let (head_dim, rel_dim) = (flash_window::FW_HEAD_DIM, flash_window::FW_REL_DIM);
        if !self.backend.resident_launches_active()
            || rows == 0 || rows > flash_window::FW_SPLIT_ROWS_MAX
            || heads == 0 || heads > flash_window::FW_HEADS_MAX
            || kv_heads == 0 || kv_heads > heads || !heads.is_multiple_of(kv_heads)
            || heads / kv_heads > flash_window::FW_SPLIT_GROUP_MAX
            || view == 0 || view > flash_window::FW_VIEW_MAX
            || extent == 0 || extent > flash_window::FW_EXTENT_MAX
            || s.row_pos_live.len() != rows
            || (!s.planes.is_empty() && (s.planes.len() != rows || s.delta.is_some())) {
            return Ok(None);
        }
        let row_words = kv_heads * (head_dim / 2);
        let multi_straight = !s.planes.is_empty() && !s.is_ring;
        if a.query.words != rows * heads * head_dim || a.gate.words != rows * heads * head_dim
            || (!multi_straight && (a.keys.words != view * row_words || a.values.words != a.keys.words))
            || a.moved.words != rows * heads * rel_dim || a.sink.words != rows * heads
            || a.steps.words != 2 * rows || a.rel_table.len() != rel_dim * extent
            || !(a.query_scale.is_empty() || a.query_scale.len() == heads * head_dim)
            || !(a.moved_scale.is_empty() || a.moved_scale.len() == heads * rel_dim)
            || a.position.is_some_and(|p| p.words != rows)
            || !(a.query_bound > 0.0 && a.moved_bound > 0.0) {
            return Ok(None);
        }
        let contig = [a.query, a.moved, a.sink, a.gate, a.steps];
        for v in contig.iter().chain(a.position.iter()) {
            if v.vector.is_segmented() || v.vector.len() != v.words { return Ok(None); }
        }
        if a.keys.vector.len() < a.keys.words || a.values.vector.len() < a.values.words {
            return Ok(None);
        }
        if (a.query.vector.device_address() | a.keys.vector.device_address() | a.values.vector.device_address()
            | a.moved.vector.device_address() | a.gate.vector.device_address()) & 15 != 0 {
            return Ok(None);
        }
        if let Some((dk, dv)) = s.delta {
            let need_d = rows * row_words;
            for d in [dk, dv] {
                if d.words != need_d || d.vector.is_segmented() || d.vector.len() != d.words || d.vector.device_address() & 15 != 0 {
                    return Ok(None);
                }
            }
        }
        for (chat, (key, value)) in s.planes.iter().enumerate() {
            let need_words = if multi_straight {
                let (pos, live) = s.row_pos_live[chat];
                if live > 0 { (pos as usize + 1) * row_words } else { row_words }
            } else {
                view * row_words
            };
            for plane in [key, value] {
                if plane.words < need_words || plane.vector.len() < need_words || plane.vector.device_address() & 15 != 0 {
                    return Ok(None);
                }
            }
        }
        for v in contig.iter().chain([a.keys, a.values].iter()).chain(a.position.iter()) { self.ready(v)?; }
        if let Some((dk, dv)) = s.delta { self.ready(dk)?; self.ready(dv)?; }
        for (key, value) in s.planes { self.ready(key)?; self.ready(value)?; }
        let planes = s.planes.iter().map(|(k, v)| (&*k.vector, &*v.vector)).collect::<Vec<_>>();
        let delta = s.delta.map(|(dk, dv)| (&*dk.vector, &*dv.vector));
        let stamp = self.cost.start(self.backend);
        let upload = |words: &[f32]| self.parameters.borrow_mut().resolve(words, None, |values| self.backend.upload_vector(values)?.into_resident_shared());
        let table = upload(a.rel_table)?.ok_or("flash window split: relative table empty")?;
        let (query_scale, moved_scale) = (upload(a.query_scale)?, upload(a.moved_scale)?);
        let clips = spark_gpu_probe::driver::OneClips {
            query_scale: query_scale.as_deref(), moved_scale: moved_scale.as_deref(),
            position: a.position.map(|p| &*p.vector),
            query_bound: a.query_bound, moved_bound: a.moved_bound,
            written_keys: None, written_values: None,
        };
        let mut output = self.backend.vector(rows * heads * head_dim)?;
        self.backend.flash_window_split_into(
            rows, heads, kv_heads, view, extent, a.scale, s.is_ring, s.first_pos, s.row_pos_live,
            &a.query.vector, &a.keys.vector, &a.values.vector, delta, &a.moved.vector, &table,
            &a.sink.vector, &a.gate.vector, &a.steps.vector, clips, &planes, &mut output,
        )?;
        let value = self.value(output, rows * heads * head_dim)?;
        self.cost.record(|| "window_split".into(), stamp, self.backend, (a.rel_table.len() + a.query_scale.len() + a.moved_scale.len()) * 4);
        Ok(Some(value))
    }

    /// `Compute::window_rows`: the tensor-core form of `FLASH_WINDOW_ATTN` with the clips inside
    /// (`Backend::flash_window_attn_into` with clips). None where the tensor-core form or its operands do not fit.
    pub(super) fn record_window_rows(&self, a: &WindowOne<'_, Value<'b, 'd>>, first_position: u64) -> Result<Option<Value<'b, 'd>>> {
        use spark_gpu_probe::flash_window;
        let (rows,heads,kv_heads,view,extent)=(a.rows,a.heads,a.kv_heads,a.view,a.extent);
        let (head_dim, rel_dim) = (flash_window::FW_HEAD_DIM, flash_window::FW_REL_DIM);
        if rows < flash_window::FW_ROWS_MIN || !self.backend.resident_launches_active()
            || !flash_window::flash_window_shape_admitted(rows, heads, kv_heads, head_dim, rel_dim, view, extent) {
            return Ok(None);
        }
        if a.query.words != rows * heads * head_dim || a.gate.words != rows * heads * head_dim || a.moved.words != rows * heads * rel_dim
            || a.sink.words != rows * heads || a.steps.words != 2 * rows || a.rel_table.len() != rel_dim * extent
            || !(a.query_scale.is_empty() || a.query_scale.len() == heads * head_dim)
            || !(a.moved_scale.is_empty() || a.moved_scale.len() == heads * rel_dim) || a.position.is_some_and(|p| p.words != rows)
            || !(a.query_bound > 0.0 && a.moved_bound > 0.0) || a.written.is_some() {
            return Ok(None);
        }
        let operands = [a.query, a.moved, a.sink, a.gate, a.steps];
        for v in operands.iter().chain(a.position.iter()) {
            if v.vector.is_segmented() { return Ok(None); }
        }
        let kv_words = view * kv_heads * (head_dim / 2);
        if a.keys.words != kv_words || a.values.words != kv_words || a.keys.vector.len() < kv_words || a.values.vector.len() < kv_words {
            return Ok(None);
        }
        if (a.query.vector.device_address() | a.keys.vector.device_address() | a.values.vector.device_address()
            | a.moved.vector.device_address() | a.gate.vector.device_address()) & 15 != 0 {
            return Ok(None);
        }
        for v in operands.iter().chain([a.keys, a.values].iter()).chain(a.position.iter()) { self.ready(v)?; }
        let stamp = self.cost.start(self.backend);
        let upload = |words: &[f32]| self.parameters.borrow_mut().resolve(words, None, |values| self.backend.upload_vector(values)?.into_resident_shared());
        let table = upload(a.rel_table)?.ok_or("flash window: relative table empty")?;
        let (query_scale, moved_scale) = (upload(a.query_scale)?, upload(a.moved_scale)?);
        let clips = spark_gpu_probe::driver::OneClips {
            query_scale: query_scale.as_deref(), moved_scale: moved_scale.as_deref(), position: a.position.map(|p| &*p.vector),
            query_bound: a.query_bound, moved_bound: a.moved_bound,
            written_keys: None, written_values: None,
        };
        let mut output = self.backend.vector(rows * heads * head_dim)?;
        self.backend.flash_window_attn_into(rows, heads, kv_heads, head_dim, rel_dim, view, extent, a.scale, first_position,
            &a.query.vector, &a.keys.vector, &a.values.vector, &a.moved.vector, &table, &a.sink.vector, &a.gate.vector, &a.steps.vector,
            Some(clips), &mut output)?;
        let value = self.value(output, rows * heads * head_dim)?;
        self.cost.record(|| "window_rows".into(), stamp, self.backend, (a.rel_table.len() + a.query_scale.len() + a.moved_scale.len()) * 4);
        Ok(Some(value))
    }

    /// `Compute::conv_finish` as one recorded arms launch (`Backend::conv_finish_into`), in place in the ring and the
    /// plane as `state_fill` writes them: in stream order, with no wait for the device.
    pub(super) fn record_conv_finish(&self, a: &ConvFinish<'_, Value<'b, 'd>>, ring: &mut Value<'b, 'd>, plane: &mut Value<'b, 'd>)
        -> Result<Option<Value<'b, 'd>>> {
        let s = a.shape;
        let (heads, taps) = (s.heads as usize, s.taps as usize);
        let width = heads * s.dim as usize;
        if !self.backend.resident_launches_active() || s.dim as usize != spark_gpu_probe::block_ops::SUM_THREADS || heads == 0 || taps == 0
            || a.product.words != width || ring.words != width * taps || a.fixed.len() != width * taps || a.step.words != 1
            || a.made.is_some_and(|m| m.words != heads * taps) || a.plane_word.checked_add(width / 2).is_none_or(|end| end > plane.words) {
            return Ok(None);
        }
        for v in [a.product, a.step, &*ring].into_iter().chain(a.made) {
            if v.vector.is_segmented() || v.vector.len() != v.words { return Ok(None); }
        }
        if plane.vector.len() < plane.words || !plane.vector.writable_from(a.plane_word) { return Ok(None); }
        if Rc::get_mut(&mut ring.vector).is_none() || Rc::get_mut(&mut plane.vector).is_none() { return Ok(None); }
        for v in [a.product, a.step, &*ring, &*plane].into_iter().chain(a.made) { self.ready(v)?; }
        let stamp = self.cost.start(self.backend);
        let fixed = self.parameters.borrow_mut().resolve(a.fixed, None, |values| self.backend.upload_vector(values)?.into_resident_shared())?
            .ok_or("conv finish: fixed taps empty")?;
        let mut output = self.backend.vector(width)?;
        let (ring_vector, plane_vector) = (Rc::get_mut(&mut ring.vector).ok_or("conv finish: ring is shared")?, Rc::get_mut(&mut plane.vector).ok_or("conv finish: plane is shared")?);
        self.backend.conv_finish_into(s, a.finish, a.param, ring_vector, a.made.map(|m| &*m.vector), &a.step.vector, &fixed, &a.product.vector,
            plane_vector, a.plane_word, &mut output)?;
        let value = self.value(output, width)?;
        self.cost.record(|| "conv_finish".into(), stamp, self.backend, a.fixed.len() * 4);
        Ok(Some(value))
    }

    /// Allocate a writable `full`-word state plane under `fill_free` (`skip_fill` skips clearing in `Skip` mode
    /// while preserving `Zero`/`Poison` proof modes) and copy the committed prefix `live` into `0..live.words`.
    pub(super) fn device_state_restore(&self, live: &Value<'b, 'd>, full: usize) -> Result<Value<'b, 'd>> {
        self.ready(live)?;
        need(live.words <= full, "state restore prefix exceeds capacity")?;
        let stamp = self.cost.start(self.backend);
        let mut out = self.backend.vector(full.max(1))?;
        self.backend.skip_fill(&mut out)?;
        self.backend.copy_range(&live.vector, &mut out, 0, 0, live.words)?;
        let value = self.value(out, full)?;
        self.cost.record(|| "state_restore".into(), stamp, self.backend, 0);
        Ok(value)
    }
}
