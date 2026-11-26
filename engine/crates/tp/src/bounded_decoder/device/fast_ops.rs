//! Fast zero-stack Inkling block-op dispatch helpers for `Device` (kinds 107..=112 in `kernels-inkling`).
use super::*;
use spark_gpu_probe::fast_ops;

impl<'b, 'd> Device<'b, 'd> {
    pub(super) fn try_short_conv_prefill(
        &self, s: Shape, rows: usize, first: usize, x: [Option<&Value<'b, 'd>>; 4], w: &[f32],
    ) -> Result<Option<Value<'b, 'd>>> {
        let (heads, dim, taps) = (s.heads as usize, s.dim as usize, s.taps as usize);
        let width = heads.checked_mul(dim).ok_or("prefill convolution width overflow")?;
        if !self.loop_prefill.get() || !self.backend.resident_launches_active()
            || !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_SHORT_CONV_WIN, rows, width, taps, fast_ops::STREAM_OPS_THREADS)
            || x.iter().flatten().any(|v| v.vector.is_segmented() || v.vector.len() != v.words)
        {
            return Ok(None);
        }
        let [Some(input), token, addend, Some(ring)] = x else { return Err("prefill convolution is missing an input".into()); };
        for value in x.iter().flatten() { self.ready(value)?; }
        let stamp = self.cost.start(self.backend);
        let fixed = self.parameters.borrow_mut().resolve(w, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("prefill convolution is missing taps")?;
        let words = rows * width;
        let mut out = self.backend.vector(words)?;
        self.backend.fast_short_conv_win_into(rows, heads, dim, taps, s.cols, s.bound, &input.vector,
            token.map(|v| &*v.vector), addend.map(|v| &*v.vector), &fixed, Some((&ring.vector, first)), &mut out, None)?;
        let value = self.value(out, words)?;
        self.cost.record(|| "fast_short_conv_prefill".into(), stamp, self.backend, w.len() * 4);
        Ok(Some(value))
    }

    pub(super) fn try_prefill_kv(
        &self, conv: &crate::bounded_decoder::PrefillKv<'_, Value<'b, 'd>>, plane: &mut Value<'b, 'd>,
    ) -> Result<bool> {
        let s = conv.shape;
        let width = s.heads as usize * s.dim as usize;
        let taps = s.taps as usize;
        let reads = [Some(conv.input), Some(conv.ring), conv.made];
        if !self.loop_prefill.get() || !self.backend.resident_launches_active()
            || s.dim as usize != fast_ops::CONV_KV_DIM
            || !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_SHORT_CONV_WIN, conv.rows, width, taps, fast_ops::STREAM_OPS_THREADS)
            || !(conv.write.param.is_finite() && conv.write.param > 0.0)
            || reads.iter().flatten().any(|v| v.vector.is_segmented() || v.vector.len() != v.words)
            || plane.vector.is_segmented() || !plane.vector.writable_from(0)
        {
            return Ok(false);
        }
        let Some(target) = std::rc::Rc::get_mut(&mut plane.vector) else { return Ok(false); };
        for v in reads.iter().flatten() { self.ready(v)?; }
        let stamp = self.cost.start(self.backend);
        let fixed = self.parameters.borrow_mut().resolve(conv.fixed, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("prefill KV convolution is missing taps")?;
        self.backend.fast_short_conv_kv_into(s, conv.rows, &conv.input.vector,
            conv.made.map(|v| &*v.vector), &fixed, (&conv.ring.vector, conv.ring_first), target, conv.write)?;
        self.cost.record(|| "fast_prefill_kv".into(), stamp, self.backend, conv.fixed.len() * 4);
        Ok(true)
    }

    /// `Compute::short_conv_keep`: runs `ShortConv` directly from `inputs` and `ring` without a `window` copy,
    /// and writes the ring after each row (`[rows][taps][width]`) in the same launch.
    pub(super) fn try_short_conv_keep(
        &self, s: Shape, rows: usize, first: usize, x: [Option<&Value<'b, 'd>>; 4], w: &[f32],
    ) -> Result<Option<(Value<'b, 'd>, Value<'b, 'd>)>> {
        let (heads, dim, taps) = (s.heads as usize, s.dim as usize, s.taps as usize);
        let width = heads.checked_mul(dim).ok_or("kept convolution width overflow")?;
        let slot_ok = first < taps || fast_ops::short_conv_multi_decode(rows, taps, first).is_some();
        if self.loop_prefill.get() || !self.backend.resident_launches_active() || rows < 2 || taps < 2 || !slot_ok
            || !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_SHORT_CONV_WIN, rows, width, taps, fast_ops::STREAM_OPS_THREADS)
            || x.iter().flatten().any(|v| v.vector.is_segmented() || v.vector.len() != v.words)
        {
            return Ok(None);
        }
        let [Some(inputs), token, addend, Some(ring)] = x else { return Err("kept convolution is missing inputs or ring".into()); };
        for value in x.iter().flatten() { self.ready(value)?; }
        let stamp = self.cost.start(self.backend);
        let fixed = self.parameters.borrow_mut().resolve(w, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("kept convolution is missing taps")?;
        let (words, kept) = (rows * width, rows * taps * width);
        let (mut out, mut every) = (self.backend.vector(words)?, self.backend.vector(kept)?);
        self.backend.fast_short_conv_win_into(rows, heads, dim, taps, s.cols, s.bound, &inputs.vector,
            token.map(|v| &*v.vector), addend.map(|v| &*v.vector), &fixed, Some((&ring.vector, first)), &mut out, Some((&mut every, first)))?;
        let (value, every) = (self.value(out, words)?, self.value(every, kept)?);
        self.cost.record(|| "fast_short_conv_keep".into(), stamp, self.backend, w.len() * 4);
        Ok(Some((value, every)))
    }

    pub(super) fn try_kv_finish_pair(&self, p: &KvFinishPair<'_, Value<'b, 'd>>) -> Result<Option<[Value<'b, 'd>; 4]>> {
        if self.loop_prefill.get() || !self.backend.resident_launches_active()
            || !fast_ops::kv_finish_admitted(p.rows, p.kv_heads, p.taps, fast_ops::KV_FINISH_THREADS)
            || p.first_slot >= p.taps
        {
            return Ok(None);
        }
        let all_in = [Some(p.key_in), Some(p.key_ring), p.key_made, Some(p.val_in), Some(p.val_ring), p.val_made];
        if all_in.iter().flatten().any(|v| v.vector.is_segmented() || v.vector.len() != v.words) {
            return Ok(None);
        }
        for v in all_in.iter().flatten() { self.ready(v)?; }
        let mut chats = Vec::with_capacity(p.chat_rings.len());
        for &(k, v, slot) in p.chat_rings {
            if k.vector.is_segmented() || v.vector.is_segmented() { return Ok(None); }
            self.ready(k)?;
            self.ready(v)?;
            chats.push((&*k.vector, &*v.vector, slot));
        }
        let stamp = self.cost.start(self.backend);
        let key_fixed = self.parameters.borrow_mut().resolve(p.key_fixed, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("kv finish pair: missing key taps")?;
        let val_fixed = self.parameters.borrow_mut().resolve(p.val_fixed, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("kv finish pair: missing value taps")?;
        let width = p.kv_heads * fast_ops::KV_FINISH_HEAD_DIM;
        let (pack_words, every_words) = (p.rows * (width / 2), p.rows * p.taps * width);
        let (mut kp, mut ke) = (self.backend.vector(pack_words)?, self.backend.vector(every_words)?);
        let (mut vp, mut ve) = (self.backend.vector(pack_words)?, self.backend.vector(every_words)?);
        self.backend.fast_kv_finish_pair_into(
            p.rows, p.kv_heads, p.taps, p.first_slot, p.key_bound, p.val_eps, p.key_scale, p.val_scale,
            &p.key_in.vector, &p.key_ring.vector, &key_fixed, p.key_made.map(|v| &*v.vector), &mut kp, Some(&mut ke),
            &p.val_in.vector, &p.val_ring.vector, &val_fixed, p.val_made.map(|v| &*v.vector), &mut vp, Some(&mut ve), &chats,
        )?;
        let out = [self.value(kp, pack_words)?, self.value(ke, every_words)?, self.value(vp, pack_words)?, self.value(ve, every_words)?];
        self.cost.record(|| "fast_kv_finish_pair".into(), stamp, self.backend, (p.key_fixed.len() + p.val_fixed.len()) * 4);
        Ok(Some(out))
    }

    pub(super) fn try_router_pick_compact(
        &self, s: Shape, rows: usize, logits: &Value<'b, 'd>, bias: &[f32],
    ) -> Result<Option<(Value<'b, 'd>, Value<'b, 'd>)>> {
        if self.loop_prefill.get() || !self.backend.resident_launches_active() || s.op != Op::RouterPick {
            return Ok(None);
        }
        let (experts, picks) = (s.rows as usize, s.cols as usize);
        if !fast_ops::router_pick_compact_admitted(rows, experts, picks, fast_ops::ROUTER_PICK_COMPACT_THREADS)
            || logits.words != rows * experts || logits.vector.is_segmented() || logits.vector.len() != logits.words
            || bias.len() != experts
        {
            return Ok(None);
        }
        self.ready(logits)?;
        let stamp = self.cost.start(self.backend);
        let bias_dev = self.parameters.borrow_mut().resolve(bias, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("fast router pick compact: missing bias")?;
        if bias_dev.is_segmented() { return Ok(None); }
        let (route_words, compact_words) = (rows * 2 * picks, rows * picks * fast_ops::ROUTER_PICK_COMPACT_ENTRY_WORDS);
        let (mut route, mut compact) = (self.backend.vector(route_words)?, self.backend.vector(compact_words)?);
        self.backend.fast_router_pick_compact_into(rows, experts, picks, s.taps != 0, s.bound, &logits.vector, &bias_dev, &mut route, &mut compact)?;
        let res = (self.value(route, route_words)?, self.value(compact, compact_words)?);
        self.cost.record(|| "fast_router_pick_compact".into(), stamp, self.backend, bias.len() * 4);
        Ok(Some(res))
    }

    pub(super) fn try_route_fold_conv(&self, p: &RouteFoldConv<'_, Value<'b, 'd>>) -> Result<Option<(Value<'b, 'd>, Value<'b, 'd>)>> {
        if self.loop_prefill.get() || !self.backend.resident_launches_active()
            || !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD_CONV, p.rows, p.dim, p.picks, fast_ops::ROUTE_FOLD_CONV_THREADS)
            || !(2..=fast_ops::ROUTE_FOLD_CONV_TAPS_MAX).contains(&p.taps) || p.first_slot >= p.taps
            || p.weight_offset.checked_add(p.picks).is_none_or(|end| end > p.weight_stride)
            || (p.half && !p.dim.is_multiple_of(2))
        {
            return Ok(None);
        }
        let all_in = [p.route, p.downs, p.residual, p.ring];
        if all_in.iter().any(|v| v.vector.is_segmented() || v.vector.len() != v.words) {
            return Ok(None);
        }
        for v in all_in { self.ready(v)?; }
        let mut chats = Vec::with_capacity(p.chat_rings.len());
        for &(ring, slot) in p.chat_rings {
            if ring.vector.is_segmented() { return Ok(None); }
            self.ready(ring)?;
            chats.push((&*ring.vector, slot));
        }
        let stamp = self.cost.start(self.backend);
        let fixed = self.parameters.borrow_mut().resolve(p.fixed, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("route fold conv: missing fixed taps")?;
        let (out_words, every_words) = (p.rows * p.dim, p.rows * p.taps * p.dim);
        let (mut out, mut every) = (self.backend.vector(out_words)?, self.backend.vector(every_words)?);
        self.backend.fast_route_fold_conv_form_into(
            p.rows, p.picks, p.dim, p.weight_stride, p.weight_offset, p.taps, p.first_slot, p.bound,
            &p.route.vector, &p.downs.vector, &p.residual.vector, &fixed, &p.ring.vector, &mut out, Some(&mut every), &chats, p.half,
        )?;
        let res = (self.value(out, out_words)?, self.value(every, every_words)?);
        self.cost.record(|| "fast_route_fold_conv".into(), stamp, self.backend, p.fixed.len() * 4);
        Ok(Some(res))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_mix_one(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, transpose: bool, keep_normed: bool,
        queries: &[f32], list: &[&Value<'b, 'd>], stream: &Value<'b, 'd>,
    ) -> Result<Option<(Option<Value<'b, 'd>>, Value<'b, 'd>)>> {
        if !self.backend.resident_launches_active() { return Ok(None); }
        let Some(asked) = queries.len().checked_div(dim) else { return Ok(None); };
        let kept = list.len().saturating_add(1);
        let row_words = rows.saturating_mul(dim);
        if !queries.len().is_multiple_of(dim)
            || !fast_ops::mix_one_admitted(rows, asked, kept, dim)
            || stream.words != row_words || stream.vector.is_segmented() || stream.vector.len() != row_words
            || stream.vector.device_address() & 15 != 0
            || list.iter().any(|v| v.words != row_words || v.vector.is_segmented() || v.vector.len() != row_words || v.vector.device_address() & 15 != 0)
        {
            return Ok(None);
        }
        for value in list { self.ready(value)?; }
        self.ready(stream)?;
        let stamp = self.cost.start(self.backend);
        let q_dev = self.parameters.borrow_mut().resolve(queries, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("fast mix one has no queries")?;
        if q_dev.is_segmented() || q_dev.device_address() & 15 != 0 { return Ok(None); }
        let refs: Vec<&DeviceVector<'d>> = list.iter().map(|v| &*v.vector).collect();
        let (normed, out) = self.backend.fast_mix_rows(rows, dim, scale, eps, transpose, keep_normed, &q_dev, &refs, &stream.vector)?;
        let normed = match normed { Some(v) => Some(self.value(v, row_words)?), None => None };
        let words = out.len();
        let out = self.value(out, words)?;
        self.cost.record(|| "fast_mix_one".into(), stamp, self.backend, 0);
        Ok(Some((normed, out)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_mix_one_normed(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, post_eps: f32,
        transpose: bool, keep_normed: bool, keep_mixed: bool,
        queries: &[f32], list: &[&Value<'b, 'd>], stream: &Value<'b, 'd>,
        post_weight: Option<&[f32]>, post_weight_per_query: bool,
    ) -> Result<Option<(Option<Value<'b, 'd>>, Option<Value<'b, 'd>>, Value<'b, 'd>)>> {
        if !self.backend.resident_launches_active() { return Ok(None); }
        let Some(asked) = queries.len().checked_div(dim) else { return Ok(None); };
        let kept = list.len().saturating_add(1);
        let row_words = rows.saturating_mul(dim);
        if !queries.len().is_multiple_of(dim)
            || !fast_ops::mix_one_admitted(rows, asked, kept, dim)
            || !fast_ops::mix_list_norm_admitted(dim, fast_ops::MIX_BLEND_THREADS)
            || !(post_eps > 0.0 && post_eps.is_finite())
            || stream.words != row_words || stream.vector.is_segmented() || stream.vector.len() != row_words
            || stream.vector.device_address() & 15 != 0
            || list.iter().any(|v| v.words != row_words || v.vector.is_segmented() || v.vector.len() != row_words || v.vector.device_address() & 15 != 0)
        {
            return Ok(None);
        }
        if let Some(pw) = post_weight {
            let need_w = if post_weight_per_query { asked.saturating_mul(dim) } else { dim };
            if pw.len() != need_w { return Ok(None); }
        }
        for value in list { self.ready(value)?; }
        self.ready(stream)?;
        let stamp = self.cost.start(self.backend);
        let q_dev = self.parameters.borrow_mut().resolve(queries, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("fast mix one has no queries")?;
        if q_dev.is_segmented() || q_dev.device_address() & 15 != 0 { return Ok(None); }
        let pw_dev = match post_weight {
            Some(pw) => {
                let d = self.parameters.borrow_mut().resolve(pw, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
                    .ok_or("fast mix one has no post weight")?;
                if d.is_segmented() { return Ok(None); }
                Some(d)
            }
            None => None,
        };
        let refs: Vec<&DeviceVector<'d>> = list.iter().map(|v| &*v.vector).collect();
        let (normed, mixed, post) = self.backend.fast_mix_rows_normed(
            rows, dim, scale, eps, post_eps, transpose, keep_normed, keep_mixed,
            &q_dev, &refs, &stream.vector, pw_dev.as_deref(), post_weight_per_query,
        )?;
        let normed = match normed { Some(v) => Some(self.value(v, row_words)?), None => None };
        let out_words = post.len();
        let mixed = match mixed { Some(v) => Some(self.value(v, out_words)?), None => None };
        let post = self.value(post, out_words)?;
        self.cost.record(|| "fast_mix_one_normed".into(), stamp, self.backend, 0);
        Ok(Some((normed, mixed, post)))
    }

    pub(super) fn try_gate_blend_norm(
        &self, rows: usize, dim: usize, eps: f32,
        entries: &[(&Value<'b, 'd>, &Value<'b, 'd>, &Value<'b, 'd>, Option<&[f32]>)],
    ) -> Result<Option<Vec<Value<'b, 'd>>>> {
        if !self.backend.resident_launches_active()
            || !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_GATE_BLEND_NORM, rows, dim, entries.len(), fast_ops::gate_blend_norm_threads(dim))
            || !(eps > 0.0 && eps.is_finite())
        {
            return Ok(None);
        }
        let g_words = rows.saturating_mul(2).saturating_mul(dim);
        let v_words = rows.saturating_mul(dim);
        for &(g, k, m, wt) in entries {
            if g.words != g_words || k.words != v_words || m.words != v_words
                || g.vector.is_segmented() || k.vector.is_segmented() || m.vector.is_segmented()
                || g.vector.len() != g_words || k.vector.len() != v_words || m.vector.len() != v_words
                || wt.is_some_and(|w| w.len() != dim)
            {
                return Ok(None);
            }
        }
        for &(g, k, m, _) in entries {
            self.ready(g)?;
            self.ready(k)?;
            self.ready(m)?;
        }
        let stamp = self.cost.start(self.backend);
        let mut w_devs: Vec<Option<Rc<DeviceVector<'d>>>> = Vec::with_capacity(entries.len());
        for &(_, _, _, wt) in entries {
            let d = match wt {
                Some(w) => {
                    let dev = self.parameters.borrow_mut().resolve(w, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
                        .ok_or("fast gate blend norm has no weight")?;
                    if dev.is_segmented() { return Ok(None); }
                    Some(dev)
                }
                None => None,
            };
            w_devs.push(d);
        }
        let refs: Vec<_> = entries.iter().zip(&w_devs)
            .map(|(&(g, k, m, _), w)| (&*g.vector, &*k.vector, &*m.vector, w.as_deref()))
            .collect();
        let outs = self.backend.fast_gate_blend_norm(rows, dim, eps, &refs)?;
        let values = outs.into_iter().map(|v| self.value(v, v_words)).collect::<Result<Vec<_>>>()?;
        self.cost.record(|| "fast_gate_blend_norm".into(), stamp, self.backend, 0);
        Ok(Some(values))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_block_ring_delta(
        &self, s: Shape, rows: usize, shared: u32, x: [&Value<'b, 'd>; 3],
        ring: &Value<'b, 'd>, delta: &Value<'b, 'd>, ring_rows: usize, start_slot: usize, prefix: usize,
    ) -> Result<Option<Value<'b, 'd>>> {
        if !self.backend.resident_queue_active() { return Ok(None); }
        for v in x.into_iter().chain([ring, delta]) { self.ready(v)?; }
        let stamp = self.cost.start(self.backend);
        let (lengths, _, units, words) = spark_gpu_probe::block_rows::lengths(s, rows, shared)
            .ok_or("invalid ring-delta attention extents")?;
        let row_words = (s.heads / s.taps) as usize * (s.dim as usize / 2);
        let delta_rows = (s.rows as usize).checked_sub(prefix).ok_or("ring-delta prefix exceeds view")?;
        need(x[0].words == lengths[0] && x[1].words == lengths[2] && x[2].words == lengths[3]
            && ring.words == ring_rows * row_words && delta.words == delta_rows * row_words,
            "ring-delta attention input extent differs")?;
        let mut output = self.backend.vector(units * words)?;
        if !self.backend.block_ring_delta_into(s, rows, shared, x.map(|v| &*v.vector),
            &ring.vector, &delta.vector, ring_rows, start_slot, prefix, &mut output)? {
            return Ok(None);
        }
        let value = self.value(output, units * words)?;
        self.cost.record(|| format!("rows_{:?}_ring_delta", s.op), stamp, self.backend, 0);
        Ok(Some(value))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_route_fold_strided(
        &self, rows: usize, picks: usize, dim: usize, weight_stride: usize, weight_offset: usize,
        route: &Value<'b, 'd>, downs: &Value<'b, 'd>,
    ) -> Result<Option<Value<'b, 'd>>> {
        if !self.backend.resident_launches_active()
            || !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD, rows, dim, picks, fast_ops::STREAM_OPS_THREADS)
            || weight_offset.checked_add(picks).is_none_or(|end| end > weight_stride)
            || route.vector.is_segmented() || downs.vector.is_segmented()
            || route.vector.len() != rows * weight_stride || downs.vector.len() != rows * picks * dim
        {
            return Ok(None);
        }
        self.ready(route)?;
        self.ready(downs)?;
        let stamp = self.cost.start(self.backend);
        let words = rows * dim;
        let mut out = self.backend.vector(words)?;
        self.backend.fast_route_fold_strided_into(rows, picks, dim, weight_stride, weight_offset, &route.vector, &downs.vector, &mut out)?;
        let value = self.value(out, words)?;
        self.cost.record(|| "fast_route_fold_strided".into(), stamp, self.backend, 0);
        Ok(Some(value))
    }

    pub(super) fn try_mix_prefill(
        &self, rows: usize, dim: usize, scale: f32, transpose: bool,
        queries: &[f32], list: &[&Value<'b, 'd>],
    ) -> Result<Option<Value<'b, 'd>>> {
        if !self.loop_prefill.get() || rows < 2 || !self.backend.resident_launches_active() {
            return Ok(None);
        }
        let Some(asked) = queries.len().checked_div(dim) else { return Ok(None); };
        let kept = list.len();
        if !queries.len().is_multiple_of(dim)
            || !fast_ops::row_dots_admitted(rows, asked, kept, dim, fast_ops::ROW_DOTS_THREADS)
            || !fast_ops::mix_blend_admitted(rows, asked, kept, dim, fast_ops::MIX_BLEND_THREADS)
            || list.iter().any(|v| v.vector.is_segmented() || v.vector.len() != rows * dim)
        {
            return Ok(None);
        }
        for value in list { self.ready(value)?; }
        let stamp = self.cost.start(self.backend);
        let queries = self.parameters.borrow_mut().resolve(queries, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("prefill mix has no queries")?;
        let list = list.iter().map(|v| &*v.vector).collect::<Vec<_>>();
        let out = self.backend.fast_mix_list(rows, dim, scale, transpose, &queries, &list)?;
        let words = out.len();
        let value = self.value(out, words)?;
        self.cost.record(|| "fast_mix_list".into(), stamp, self.backend, 0);
        Ok(Some(value))
    }

    /// `Compute::mix_prefill_normed` on one launch (`Backend::fast_mix_list_normed`): where `try_mix_prefill` takes the
    /// mix and `fast_norm` takes the rounded norm of its rows (`try_fast_operation`), so each word is the word of those two.
    pub(super) fn try_mix_prefill_normed(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, queries: &[f32], list: &[&Value<'b, 'd>],
    ) -> Result<Option<(Value<'b, 'd>, Value<'b, 'd>)>> {
        if !self.loop_prefill.get() || rows < 2 || !self.backend.resident_launches_active() {
            return Ok(None);
        }
        let Some(asked) = queries.len().checked_div(dim) else { return Ok(None); };
        let kept = list.len();
        if !queries.len().is_multiple_of(dim)
            || !fast_ops::row_dots_admitted(rows, asked, kept, dim, fast_ops::ROW_DOTS_THREADS)
            || !fast_ops::mix_blend_admitted(rows, asked, kept, dim, fast_ops::MIX_BLEND_THREADS)
            || !fast_ops::mix_list_norm_admitted(dim, fast_ops::MIX_BLEND_THREADS)
            || !fast_ops::norm_admitted(rows * asked, dim, true, fast_ops::NORM_THREADS)
            || !(eps > 0.0 && eps.is_finite())
            || list.iter().any(|v| v.vector.is_segmented() || v.vector.len() != rows * dim)
        {
            return Ok(None);
        }
        for value in list { self.ready(value)?; }
        let stamp = self.cost.start(self.backend);
        let queries = self.parameters.borrow_mut().resolve(queries, None, |v| self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("prefill mix has no queries")?;
        let list = list.iter().map(|v| &*v.vector).collect::<Vec<_>>();
        let (out, normed) = self.backend.fast_mix_list_normed(rows, dim, scale, eps, &queries, &list)?;
        let words = out.len();
        let both = (self.value(out, words)?, self.value(normed, words)?);
        self.cost.record(|| "fast_mix_list_normed".into(), stamp, self.backend, 0);
        Ok(Some(both))
    }

    /// Try to run a single-call block arm (`Device::operation_identities`) via the zero-stack fast kernels.
    pub(super) fn try_fast_operation(
        &self, s: Shape, input: [Option<&DeviceVector<'d>>; 4], weight: Option<&DeviceVector<'d>>, w: &[f32],
    ) -> Result<Option<DeviceVector<'d>>> {
        if !self.backend.resident_launches_active() { return Ok(None); }
        let ok = |v: &DeviceVector<'d>| !v.is_segmented();
        match s.op {
            Op::PlainNorm => {
                let (rows, dim) = (s.heads as usize, s.dim as usize);
                let Some(x0) = input[0].filter(|v| ok(v)) else { return Ok(None); };
                if !fast_ops::norm_admitted(rows, dim, false, fast_ops::NORM_THREADS) { return Ok(None); }
                let mut out = self.backend.vector(rows * dim)?;
                self.backend.fast_norm_into(rows, dim, false, s.epsilon, self.loop_prefill.get(), x0, None, &mut out)?;
                Ok(Some(out))
            }
            Op::RoundedNorm => {
                let (rows, dim) = (s.heads as usize, s.dim as usize);
                let Some(x0) = input[0].filter(|v| ok(v)) else { return Ok(None); };
                if weight.is_some_and(|wt| !ok(wt)) || !fast_ops::norm_admitted(rows, dim, true, fast_ops::NORM_THREADS) { return Ok(None); }
                let mut out = self.backend.vector(rows * dim)?;
                self.backend.fast_norm_into(rows, dim, true, s.epsilon, self.loop_prefill.get(), x0, weight, &mut out)?;
                Ok(Some(out))
            }
            Op::CappedGate => {
                let (picks, dim) = (s.rows as usize, s.dim as usize);
                let Some(x0) = input[0].filter(|v| ok(v)) else { return Ok(None); };
                if w.len() != 2 || !fast_ops::capped_gate_admitted(picks, dim, fast_ops::CAPPED_GATE_THREADS) { return Ok(None); }
                let mut out = self.backend.vector(picks * dim)?;
                self.backend.fast_capped_gate_into(picks, dim, s.epsilon, s.bound, w[0], w[1], self.loop_prefill.get(), x0, &mut out)?;
                Ok(Some(out))
            }
            Op::RouterPick => {
                let (experts, picks) = (s.rows as usize, s.cols as usize);
                let (Some(x0), Some(wt)) = (input[0].filter(|v| ok(v)), weight.filter(|v| ok(v))) else { return Ok(None); };
                if !fast_ops::router_pick_admitted(1, experts, picks, fast_ops::ROUTER_PICK_THREADS) { return Ok(None); }
                let mut out = self.backend.vector(2 * picks)?;
                self.backend.fast_router_pick_into(1, experts, picks, s.taps != 0, s.bound, x0, wt, &mut out)?;
                Ok(Some(out))
            }
            Op::RouteFold => {
                let (picks, dim) = (s.rows as usize, s.dim as usize);
                let (Some(x0), Some(x1)) = (input[0].filter(|v| ok(v)), input[1].filter(|v| ok(v))) else { return Ok(None); };
                if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD, 1, dim, picks, fast_ops::STREAM_OPS_THREADS) { return Ok(None); }
                let mut out = self.backend.vector(dim)?;
                self.backend.fast_route_fold_into(1, picks, dim, x0, x1, &mut out)?;
                Ok(Some(out))
            }
            Op::GateBlend => {
                let dim = s.dim as usize;
                let (Some(x0), Some(x1), Some(x2)) = (input[0].filter(|v| ok(v)), input[1].filter(|v| ok(v)), input[2].filter(|v| ok(v))) else { return Ok(None); };
                if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_GATE_BLEND, 1, dim, 0, fast_ops::STREAM_OPS_THREADS) { return Ok(None); }
                let mut out = self.backend.vector(dim)?;
                self.backend.fast_gate_blend_into(1, dim, x0, x1, x2, &mut out)?;
                Ok(Some(out))
            }
            Op::MixBlend => {
                let (asked, kept, dim) = (s.heads as usize, s.cols as usize, s.dim as usize);
                let (Some(x0), Some(x1)) = (input[0].filter(|v| ok(v)), input[1].filter(|v| ok(v))) else { return Ok(None); };
                if !fast_ops::mix_blend_admitted(1, asked, kept, dim, fast_ops::MIX_BLEND_THREADS) { return Ok(None); }
                let mut out = self.backend.vector(asked * dim)?;
                self.backend.fast_mix_blend_into(1, asked, kept, dim, s.bound, false, x0, x1, &mut out)?;
                Ok(Some(out))
            }
            _ => Ok(None),
        }
    }

    /// Try to run a multi-row `block_rows` launch via the zero-stack fast kernels.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_fast_block_rows(
        &self, s: Shape, rows: usize, shared: u32, input: [Option<&DeviceVector<'d>>; 4],
        weight: Option<&DeviceVector<'d>>, w: &[f32], output: &mut DeviceVector<'d>,
    ) -> Result<bool> {
        use spark_gpu_probe::block_rows;
        if !self.backend.resident_launches_active() { return Ok(false); }
        let ok = |v: &DeviceVector<'d>| !v.is_segmented();
        match s.op {
            Op::HeadClip if shared == 0 && rows > 1 => {
                let (heads, dim) = (s.heads as usize, s.dim as usize);
                let Some(x0) = input[0].filter(|v| ok(v)) else { return Ok(false); };
                if !fast_ops::head_clip_admitted(rows.saturating_mul(heads), dim, fast_ops::NORM_THREADS)
                    || !s.bound.is_finite() || input[1].is_some_and(|v| !ok(v)) || weight.is_some_and(|v| !ok(v))
                {
                    return Ok(false);
                }
                self.backend.fast_head_clip_into(s, rows, x0, weight, input[1], output)?;
                Ok(true)
            }
            Op::MixBlend if shared == block_rows::LIST => {
                let (asked, kept, dim) = (s.heads as usize, s.cols as usize, s.dim as usize);
                let (Some(x0), Some(x1)) = (input[0].filter(|v| ok(v)), input[1].filter(|v| ok(v))) else { return Ok(false); };
                if !fast_ops::mix_blend_admitted(rows, asked, kept, dim, fast_ops::MIX_BLEND_THREADS) { return Ok(false); }
                self.backend.fast_mix_blend_into(rows, asked, kept, dim, s.bound, false, x0, x1, output)?;
                Ok(true)
            }
            Op::RowDots if self.loop_prefill.get() && shared == block_rows::LIST => {
                let (asked, kept, k) = (s.rows as usize, s.heads as usize, s.cols as usize);
                let (Some(x0), Some(wt)) = (input[0].filter(|v| ok(v)), weight.filter(|v| ok(v))) else { return Ok(false); };
                if !fast_ops::row_dots_admitted(rows, asked, kept, k, fast_ops::ROW_DOTS_THREADS) { return Ok(false); }
                self.backend.fast_row_dots_into(rows, asked, kept, k, x0, wt, output)?;
                Ok(true)
            }
            Op::RouterPick if shared == 0 => {
                let (experts, picks) = (s.rows as usize, s.cols as usize);
                let (Some(x0), Some(wt)) = (input[0].filter(|v| ok(v)), weight.filter(|v| ok(v))) else { return Ok(false); };
                if !fast_ops::router_pick_admitted(rows, experts, picks, fast_ops::ROUTER_PICK_THREADS) { return Ok(false); }
                self.backend.fast_router_pick_into(rows, experts, picks, s.taps != 0, s.bound, x0, wt, output)?;
                Ok(true)
            }
            Op::ShortConv if shared == (block_rows::WINDOW | 8) => {
                let (heads, dim, taps) = (s.heads as usize, s.dim as usize, s.taps as usize);
                let Some(width) = heads.checked_mul(dim) else { return Ok(false); };
                let (Some(x0), Some(wt)) = (input[0].filter(|v| ok(v)), weight.filter(|v| ok(v))) else { return Ok(false); };
                if input[1].is_some_and(|v| !ok(v)) || input[2].is_some_and(|v| !ok(v)) { return Ok(false); }
                if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_SHORT_CONV_WIN, rows, width, taps, fast_ops::STREAM_OPS_THREADS) { return Ok(false); }
                self.backend.fast_short_conv_win_into(rows, heads, dim, taps, s.cols, s.bound, x0, input[1], input[2], wt, None, output, None)?;
                Ok(true)
            }
            Op::RouteFold if shared == 0 => {
                let (picks, dim) = (s.rows as usize, s.dim as usize);
                let (Some(x0), Some(x1)) = (input[0].filter(|v| ok(v)), input[1].filter(|v| ok(v))) else { return Ok(false); };
                if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD, rows, dim, picks, fast_ops::STREAM_OPS_THREADS) { return Ok(false); }
                self.backend.fast_route_fold_into(rows, picks, dim, x0, x1, output)?;
                Ok(true)
            }
            Op::GateBlend if shared == 0 => {
                let dim = s.dim as usize;
                let (Some(x0), Some(x1), Some(x2)) = (input[0].filter(|v| ok(v)), input[1].filter(|v| ok(v)), input[2].filter(|v| ok(v))) else { return Ok(false); };
                if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_GATE_BLEND, rows, dim, 0, fast_ops::STREAM_OPS_THREADS) { return Ok(false); }
                self.backend.fast_gate_blend_into(rows, dim, x0, x1, x2, output)?;
                Ok(true)
            }
            Op::CappedGate if shared == 0 => {
                let (picks, dim) = (rows.saturating_mul(s.rows as usize), s.dim as usize);
                let Some(x0) = input[0].filter(|v| ok(v)) else { return Ok(false); };
                if w.len() != 2 || !fast_ops::capped_gate_admitted(picks, dim, fast_ops::CAPPED_GATE_THREADS) { return Ok(false); }
                self.backend.fast_capped_gate_into(picks, dim, s.epsilon, s.bound, w[0], w[1], self.loop_prefill.get(), x0, output)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Direct transposed list-form `MixBlend` (`[asked][rows][dim]`), eliminating post-mix `PermuteRows` swaps.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_mix_blend_transposed(
        &self, rows: usize, asked: usize, kept: usize, dim: usize, scale: f32,
        scores: &Value<'b, 'd>, joined: &Value<'b, 'd>,
    ) -> Result<Option<Value<'b, 'd>>> {
        if !self.backend.resident_launches_active()
            || !fast_ops::mix_blend_admitted(rows, asked, kept, dim, fast_ops::MIX_BLEND_THREADS)
            || scores.vector.is_segmented() || joined.vector.is_segmented()
        {
            return Ok(None);
        }
        self.ready(scores)?;
        self.ready(joined)?;
        let stamp = self.cost.start(self.backend);
        let words = rows * asked * dim;
        let mut out = self.backend.vector(words)?;
        self.backend.fast_mix_blend_into(rows, asked, kept, dim, scale, true, &scores.vector, &joined.vector, &mut out)?;
        let value = self.value(out, words)?;
        self.cost.record(|| "fast_mix_blend_transposed".into(), stamp, self.backend, 0);
        Ok(Some(value))
    }
}
