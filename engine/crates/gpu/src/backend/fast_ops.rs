//! Safe host launch wrappers for the zero-stack fast Inkling block-op kernels (kinds 107..=112 in
//! `kernels-inkling`, Ruling R6h). Every extent, alignment, context ownership, and non-overlap invariant
//! is verified via `crate::fast_ops::check_*` before recording into the resident launch queue.
use super::*;
use crate::fast_ops::{self, Span};
use crate::resident_launch::{Command, Kind};

impl<'a> Backend<'a> {
    /// Fused decode mix (`fast_ops::MIX_ONE`, `rows = 1..=16`): computes the plain norm of `stream`,
    /// the sequential dot chains of `queries` against `list` and the normed stream, softmax, and blend in
    /// one `FAST_MIX_BLEND` launch without a join copy. Returns `(optional normed stream, blended output)`.
    #[allow(clippy::too_many_arguments, reason = "shape, norm parameters, layout flags and device operands")]
    pub fn fast_mix_one(
        &self, asked: usize, dim: usize, scale: f32, eps: f32,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>], stream: &DeviceVector<'_>,
    ) -> Result<(DeviceVector<'a>, DeviceVector<'a>), String> {
        if asked.checked_mul(dim) != Some(queries.len()) { return Err("fast mix one: query extent differs from shape".into()); }
        let (normed, mixed) = self.fast_mix_rows(1, dim, scale, eps, false, true, queries, list, stream)?;
        Ok((normed.ok_or("fast mix one: missing normed row")?, mixed))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn fast_mix_rows(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, transpose: bool, keep_normed: bool,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>], stream: &DeviceVector<'_>,
    ) -> Result<(Option<DeviceVector<'a>>, DeviceVector<'a>), String> {
        let (normed, mixed, _) = self.fast_mix_rows_inner(
            rows, dim, scale, eps, None, false, transpose, keep_normed, true, queries, list, stream, None, false,
        )?;
        Ok((normed, mixed.ok_or("fast mix one: missing mixed output")?))
    }

    /// Fused decode/prefill mix (`MIX_ONE`) with fused post-blend `RoundedNorm` (`post_eps`, optional `post_weight`).
    /// When `keep_mixed` is false, skips allocating/writing the un-normed blend output.
    #[allow(clippy::too_many_arguments)]
    pub fn fast_mix_rows_normed(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, post_eps: f32,
        transpose: bool, keep_normed: bool, keep_mixed: bool,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>], stream: &DeviceVector<'_>,
        post_weight: Option<&DeviceVector<'_>>, post_weight_per_query: bool,
    ) -> Result<(Option<DeviceVector<'a>>, Option<DeviceVector<'a>>, DeviceVector<'a>), String> {
        let (normed, mixed, post) = self.fast_mix_rows_inner(
            rows, dim, scale, eps, Some(post_eps), false, transpose, keep_normed, keep_mixed,
            queries, list, stream, post_weight, post_weight_per_query,
        )?;
        Ok((normed, mixed, post.ok_or("fast mix one: missing post-norm output")?))
    }

    /// `fast_mix_rows_normed` emitting `post` as packed BF16 (`MIX_ONE_POST_HALF_OUT`, `rows * asked * dim / 2` words).
    #[allow(clippy::too_many_arguments)]
    pub fn fast_mix_rows_normed_half(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, post_eps: f32,
        transpose: bool, keep_normed: bool, keep_mixed: bool,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>], stream: &DeviceVector<'_>,
        post_weight: Option<&DeviceVector<'_>>, post_weight_per_query: bool,
    ) -> Result<(Option<DeviceVector<'a>>, Option<DeviceVector<'a>>, DeviceVector<'a>), String> {
        let (normed, mixed, post) = self.fast_mix_rows_inner(
            rows, dim, scale, eps, Some(post_eps), true, transpose, keep_normed, keep_mixed,
            queries, list, stream, post_weight, post_weight_per_query,
        )?;
        Ok((normed, mixed, post.ok_or("fast mix one: missing post-norm output")?))
    }

    #[allow(clippy::too_many_arguments)]
    fn fast_mix_rows_inner(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, post_eps: Option<f32>, post_half: bool,
        transpose: bool, keep_normed: bool, keep_mixed: bool,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>], stream: &DeviceVector<'_>,
        post_weight: Option<&DeviceVector<'_>>, post_weight_per_query: bool,
    ) -> Result<(Option<DeviceVector<'a>>, Option<DeviceVector<'a>>, Option<DeviceVector<'a>>), String> {
        if self.arm_capture() { return Err("fast mix one: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        if !self.resident_launches_active() { return Err("fast mix one: launches are not recorded now".into()); }
        let asked = queries.len().checked_div(dim).ok_or("fast mix one: zero width")?;
        if !queries.len().is_multiple_of(dim) { return Err("fast mix one: query extent differs from width".into()); }
        let kept = list.len().checked_add(1).ok_or("fast mix one: kept overflow")?;
        let row_words = rows.checked_mul(dim).ok_or("fast mix one: row extent overflow")?;
        let out_words = rows.checked_mul(asked).and_then(|n| n.checked_mul(dim)).ok_or("fast mix one: output overflow")?;
        self.check_vector(queries, asked * dim)?;
        self.check_vector(stream, row_words)?;
        if queries.is_segmented() || stream.is_segmented() { return Err("fast mix one: operand must be contiguous".into()); }
        for &v in list {
            self.check_vector(v, row_words)?;
            if v.is_segmented() { return Err("fast mix one: list entry must be contiguous".into()); }
        }
        if let Some(pw) = post_weight {
            let pw_words = if post_weight_per_query { asked.checked_mul(dim).ok_or("fast mix one: weight overflow")? } else { dim };
            self.check_vector(pw, pw_words)?;
            if pw.is_segmented() { return Err("fast mix one: post-norm weight must be contiguous".into()); }
        }
        self.require_arm_kernels(&[Kind::FastMixBlend.kernel()])?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let list_spans: Vec<Span> = list.iter().map(|&v| span(v)).collect();
        let normed = if keep_normed { Some(self.vector(row_words)?) } else { None };
        let output = if keep_mixed { Some(self.vector(out_words)?) } else { None };
        let post_out = if post_eps.is_some() { Some(self.vector(if post_half { out_words / 2 } else { out_words })?) } else { None };
        fast_ops::check_mix_rows_normed_form(
            rows, asked, dim, scale, eps, post_eps, post_half, span(queries), &list_spans, span(stream),
            normed.as_ref().map(span), output.as_ref().map(span), post_out.as_ref().map(span),
            post_weight.map(span), post_weight_per_query,
        )?;
        let fused_q = fast_ops::mix_one_fused_q(rows, asked, post_eps.is_some());
        let mut command = Command::new(Kind::FastMixBlend);
        let mut flags = fast_ops::MIX_ONE | u64::from(transpose);
        if post_eps.is_some() {
            flags |= fast_ops::MIX_ONE_POST_NORM;
            if !keep_mixed { flags |= fast_ops::MIX_ONE_SKIP_MIXED; }
            if post_weight_per_query { flags |= fast_ops::MIX_ONE_POST_WEIGHTS_PER_QUERY; }
            if post_half { flags |= fast_ops::MIX_ONE_POST_HALF_OUT; }
            if fused_q { flags |= fast_ops::MIX_ONE_FUSED_Q; }
        }
        for (i, w) in [
            queries.device_address(), 0, output.as_ref().map_or(0, DeviceVector::device_address),
            rows as u64, asked as u64, kept as u64, dim as u64,
            u64::from(scale.to_bits()), flags,
        ].into_iter().enumerate() {
            command.word(i, w);
        }
        for (i, &v) in list.iter().enumerate() { command.word(9 + i, v.device_address()); }
        command.word(9 + list.len(), stream.device_address());
        command.word(fast_ops::MIX_ONE_NORMED_AT, normed.as_ref().map_or(0, DeviceVector::device_address));
        command.word(fast_ops::MIX_ONE_EPSILON_AT, u64::from(eps.to_bits()));
        if let (Some(pe), Some(po)) = (post_eps, post_out.as_ref()) {
            command.word(fast_ops::MIX_ONE_POST_OUT_AT, po.device_address());
            command.word(fast_ops::MIX_ONE_POST_WEIGHT_AT, post_weight.map_or(0, DeviceVector::device_address));
            command.word(fast_ops::MIX_ONE_POST_EPS_AT, u64::from(pe.to_bits()));
        }
        let parts = fast_ops::mix_one_parts(rows, asked, dim);
        let blocks = if fused_q {
            1
        } else {
            u32::try_from(asked * parts).map_err(|_| "fast mix one: grid overflow")?
        };
        let grid = [blocks, rows as u32, 1, fast_ops::MIX_BLEND_THREADS as u32, 0];
        self.resident_completion()?;
        if !self.record_resident_block(Kind::FastMixBlend, command, grid)? {
            return Err("fast mix one: launches are not recorded now".into());
        }
        self.resident_completion()?;
        Ok((normed, output, post_out))
    }

    /// Run the prefill mix on separate, immutable stream buffers.
    pub fn fast_mix_list(
        &self, rows: usize, dim: usize, scale: f32, transpose: bool,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>],
    ) -> Result<DeviceVector<'a>, String> {
        self.mix_list(rows, dim, scale, transpose, None, false, queries, list).map(|(output, _)| output)
    }
    /// The transposed prefill mix and the gate norm of every blended row in one launch (`fast_ops::MIX_LIST_NORM`):
    /// (the words of `fast_mix_list` with `transpose`, the words of `fast_norm_into` of them: rounded, no weight,
    /// epsilon `eps`). The norm does not read the blend words back from memory.
    pub fn fast_mix_list_normed(
        &self, rows: usize, dim: usize, scale: f32, eps: f32,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>],
    ) -> Result<(DeviceVector<'a>, DeviceVector<'a>), String> {
        if !fast_ops::mix_list_norm_admitted(dim, fast_ops::MIX_BLEND_THREADS) || !(eps > 0.0 && eps.is_finite()) {
            return Err("prefill mix with norm: width or epsilon outside the norm's envelope".into());
        }
        let (output, normed) = self.mix_list(rows, dim, scale, true, Some(eps), false, queries, list)?;
        Ok((output, normed.ok_or("prefill mix with norm made no normed rows")?))
    }
    /// `fast_mix_list_normed` emitting `normed` as packed BF16 (`MIX_LIST_NORM_HALF`, `rows * asked * dim / 2` words).
    pub fn fast_mix_list_normed_half(
        &self, rows: usize, dim: usize, scale: f32, eps: f32,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>],
    ) -> Result<(DeviceVector<'a>, DeviceVector<'a>), String> {
        if !fast_ops::mix_list_norm_admitted(dim, fast_ops::MIX_BLEND_THREADS)
            || dim > fast_ops::MIX_ONE_DIM_MAX
            || !(eps > 0.0 && eps.is_finite())
        {
            return Err("prefill mix with half norm: width or epsilon outside envelope".into());
        }
        let (output, normed) = self.mix_list(rows, dim, scale, true, Some(eps), true, queries, list)?;
        Ok((output, normed.ok_or("prefill mix with half norm made no normed rows")?))
    }
    #[allow(clippy::too_many_arguments, reason = "shape, scale, form, epsilon and the device operands")]
    fn mix_list(
        &self, rows: usize, dim: usize, scale: f32, transpose: bool, norm: Option<f32>, norm_half: bool,
        queries: &DeviceVector<'_>, list: &[&DeviceVector<'_>],
    ) -> Result<(DeviceVector<'a>, Option<DeviceVector<'a>>), String> {
        self.require_resident_stream()?;
        if !self.resident_launches_active() || self.arm_capture() {
            return Err("prefill mix requires resident recording".into());
        }
        let asked = queries.len().checked_div(dim).ok_or("prefill mix has zero width")?;
        let kept = list.len();
        if !fast_ops::row_dots_admitted(rows, asked, kept, dim, fast_ops::ROW_DOTS_THREADS)
            || !fast_ops::mix_blend_admitted(rows, asked, kept, dim, fast_ops::MIX_BLEND_THREADS)
            || !(scale > 0.0 && scale.is_finite())
        {
            return Err("prefill mix shape or scale differs".into());
        }
        for (vector, words) in list.iter().map(|&v| (v, rows * dim)).chain([(queries, asked * dim)]) {
            self.check_vector(vector, words)?;
            if vector.len() != words || vector.is_segmented() || !vector.device_address().is_multiple_of(4) {
                return Err("prefill mix requires aligned contiguous streams".into());
            }
        }
        self.require_arm_kernels(&[Kind::FastRowDots.kernel(), Kind::FastMixBlend.kernel()])?;
        let output = self.vector(rows * asked * dim)?;
        // The normed rows (`fast_ops::MIX_LIST_NORM`): a new private vector, so it meets no input and not the output.
        let normed = if norm.is_some() {
            Some(self.vector(if norm_half { rows * asked * dim / 2 } else { rows * asked * dim })?)
        } else {
            None
        };
        let norm_form = if norm.is_some() {
            fast_ops::MIX_LIST_NORM | if norm_half { fast_ops::MIX_LIST_NORM_HALF } else { 0 }
        } else {
            0
        };
        let norm_words = |mix: &mut Command| if let (Some(eps), Some(normed)) = (norm, normed.as_ref()) {
            mix.word(fast_ops::MIX_LIST_NORMED_AT, normed.device_address());
            mix.word(fast_ops::MIX_LIST_EPSILON_AT, u64::from(eps.to_bits()));
        };
        if asked <= fast_ops::MIX_DOTS_ASKED_MAX {
            // One launch: each blend block makes the dots of its query (`fast_ops::MIX_DOTS`).
            let mut mix = Command::new(Kind::FastMixBlend);
            for (i, word) in [queries.device_address(), 0, output.device_address(), rows as u64, asked as u64, kept as u64,
                dim as u64, u64::from(scale.to_bits()), u64::from(transpose) | fast_ops::MIX_DOTS | norm_form].into_iter().enumerate()
            {
                mix.word(i, word);
            }
            for (i, vector) in list.iter().enumerate() { mix.word(9 + i, vector.device_address()); }
            norm_words(&mut mix);
            let mix_grid = [(rows * asked) as u32, 1, 1, fast_ops::MIX_BLEND_THREADS as u32, 0];
            self.resident_completion()?;
            if !self.record_resident_block(Kind::FastMixBlend, mix, mix_grid)? {
                return Err("prefill mix launch is not recorded now".into());
            }
            self.resident_completion()?;
            return Ok((output, normed));
        }
        let scores = self.vector(rows * asked * kept)?;
        let mut dots = Command::new(Kind::FastRowDots);
        let mut mix = Command::new(Kind::FastMixBlend);
        for (i, word) in [0, queries.device_address(), scores.device_address(), rows as u64, asked as u64, kept as u64, dim as u64]
            .into_iter().enumerate()
        {
            dots.word(i, word);
        }
        for (i, word) in [scores.device_address(), 0, output.device_address(), rows as u64, asked as u64, kept as u64,
            dim as u64, u64::from(scale.to_bits()), u64::from(transpose) | norm_form].into_iter().enumerate()
        {
            mix.word(i, word);
        }
        for (i, vector) in list.iter().enumerate() {
            dots.word(7 + i, vector.device_address());
            mix.word(9 + i, vector.device_address());
        }
        norm_words(&mut mix);
        let dots_grid = [(rows * kept) as u32, 1, 1, fast_ops::ROW_DOTS_THREADS as u32, 0];
        let mix_grid = [(rows * asked) as u32, 1, 1, fast_ops::MIX_BLEND_THREADS as u32, 0];
        self.resident_completion()?;
        if !self.record_resident_block(Kind::FastRowDots, dots, dots_grid)?
            || !self.record_resident_block(Kind::FastMixBlend, mix, mix_grid)?
        {
            return Err("prefill mix launches are not recorded now".into());
        }
        self.resident_completion()?;
        Ok((output, normed))
    }

    #[allow(clippy::too_many_arguments, reason = "shape, scale, transpose flag and three device operands")]
    pub fn fast_mix_blend_into(
        &self, rows: usize, asked: usize, kept: usize, dim: usize, scale: f32, transpose: bool,
        scores: &DeviceVector<'_>, joined: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast mix blend: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        fast_ops::check_mix_blend(rows, asked, kept, dim, scale, span(scores), span(joined), span(output))?;
        self.check_vector(scores, rows * asked * kept)?;
        self.check_vector(joined, kept * rows * dim)?;
        self.check_vector(output, rows * asked * dim)?;
        if scores.is_segmented() || joined.is_segmented() { return Err("fast mix blend: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast mix blend: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::FastMixBlend.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastMixBlend);
        let words = [
            scores.device_address(), joined.device_address(), output.device_address(),
            rows as u64, asked as u64, kept as u64, dim as u64,
            u64::from(scale.to_bits()), u64::from(transpose),
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let blocks = u32::try_from(rows * asked * fast_ops::mix_blend_parts(dim)).map_err(|_| "fast mix blend: grid overflow")?;
        let grid = [blocks, 1, 1, fast_ops::MIX_BLEND_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastMixBlend, command, grid)? {
            return Err("fast mix blend: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    #[allow(clippy::too_many_arguments, reason = "shape and three device operands")]
    pub fn fast_row_dots_into(
        &self, rows: usize, asked: usize, kept: usize, k: usize,
        joined: &DeviceVector<'_>, queries: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast row dots: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        fast_ops::check_row_dots(rows, asked, kept, k, span(joined), span(queries), span(output))?;
        self.check_vector(joined, kept * rows * k)?;
        self.check_vector(queries, asked * k)?;
        self.check_vector(output, rows * asked * kept)?;
        if joined.is_segmented() || queries.is_segmented() { return Err("fast row dots: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast row dots: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::FastRowDots.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastRowDots);
        let words = [
            joined.device_address(), queries.device_address(), output.device_address(),
            rows as u64, asked as u64, kept as u64, k as u64,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let blocks = u32::try_from(rows * kept).map_err(|_| "fast row dots: grid overflow")?;
        let grid = [blocks, 1, 1, fast_ops::ROW_DOTS_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastRowDots, command, grid)? {
            return Err("fast row dots: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    #[allow(clippy::too_many_arguments, reason = "shape, epsilon and three device operands")]
    pub fn fast_norm_into(
        &self, rows: usize, dim: usize, rounded: bool, eps: f32, prefill: bool,
        input: &DeviceVector<'_>, weight: Option<&DeviceVector<'_>>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast norm: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        fast_ops::check_norm(rows, dim, rounded, eps, span(input), weight.map(span), span(output))?;
        self.check_vector(input, rows * dim)?;
        if let Some(w) = weight { self.check_vector(w, dim)?; if w.is_segmented() { return Err("fast norm: weight segmented".into()); } }
        self.check_vector(output, rows * dim)?;
        if input.is_segmented() { return Err("fast norm: input must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast norm: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::FastNorm.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastNorm);
        let words = [
            input.device_address(), weight.map_or(0, DeviceVector::device_address), output.device_address(),
            rows as u64, dim as u64, u64::from(rounded), u64::from(eps.to_bits()),
            0, 0, u64::from(prefill),
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let grid = [fast_ops::norm_blocks(rows, dim, rounded), 1, 1, fast_ops::NORM_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastNorm, command, grid)? {
            return Err("fast norm: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Run a prefill head clip with one warp per head.
    pub fn fast_head_clip_into(
        &self, shape: crate::block_ops::Shape, rows: usize,
        input: &DeviceVector<'_>, weight: Option<&DeviceVector<'_>>,
        factor: Option<&DeviceVector<'_>>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        let (lengths, weights, units, unit_words) = crate::block_rows::lengths(shape, rows, 0)
            .ok_or("prefill head clip shape differs")?;
        let heads = shape.heads as usize;
        let dim = shape.dim as usize;
        let total = rows * heads;
        if self.arm_capture() || !self.resident_launches_active()
            || shape.op != crate::block_ops::Op::HeadClip
            || !fast_ops::head_clip_admitted(total, dim, fast_ops::NORM_THREADS)
            || !shape.bound.is_finite()
            || !output.writable_from(0)
        {
            return Err("prefill head clip is outside the fast launch envelope".into());
        }
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let out = span(output);
        let check = |vector: &DeviceVector<'_>, words| {
            self.check_vector(vector, words)?;
            if vector.is_segmented() || !span(vector).valid(Some(words as u64 * 4)) {
                return Err("prefill head clip requires aligned contiguous operands".to_owned());
            }
            Ok(())
        };
        check(output, units * unit_words)?;
        for (operand, words) in [(Some(input), lengths[0]), (weight, weights), (factor, lengths[1])] {
            match operand {
                Some(v) if words > 0 && !out.meets(span(v)) => check(v, words)?,
                None if words == 0 => {},
                _ => return Err("prefill head clip operand is missing or overlaps output".into()),
            }
        }
        self.require_arm_kernels(&[Kind::FastNorm.kernel()])?;
        let mut command = Command::new(Kind::FastNorm);
        for (i, word) in [input.device_address(), weight.map_or(0, DeviceVector::device_address),
            output.device_address(), total as u64, dim as u64, 2, u64::from(shape.bound.to_bits()),
            heads as u64, factor.map_or(0, DeviceVector::device_address)].into_iter().enumerate()
        {
            command.word(i, word);
        }
        self.resident_completion()?;
        let grid = [fast_ops::norm_blocks(total, dim, false), 1, 1, fast_ops::NORM_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastNorm, command, grid)? {
            return Err("prefill head clip launch is not recorded now".into());
        }
        self.resident_completion()
    }

    #[allow(clippy::too_many_arguments, reason = "shape, caps, reciprocals and two device operands")]
    pub fn fast_capped_gate_into(
        &self, picks: usize, dim: usize, gate_cap: f32, up_cap: f32, inv_gate: f32, inv_up: f32, prefill: bool,
        input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast capped gate: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        fast_ops::check_capped_gate(picks, dim, gate_cap, up_cap, inv_gate, inv_up, span(input), span(output))?;
        self.check_vector(input, picks * 2 * dim)?;
        self.check_vector(output, picks * dim)?;
        if input.is_segmented() { return Err("fast capped gate: input must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast capped gate: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::FastCappedGate.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastCappedGate);
        let vector = prefill && dim.is_multiple_of(4)
            && (input.device_address() | output.device_address()).is_multiple_of(16);
        // The gate reads BF16. Reuse its two exact FP32 coefficients for wide
        // prefill work after there are enough words to pay for the table.
        let table_words = fast_ops::CAPPED_GATE_TABLE_WORDS;
        let coefficients = if vector && picks * dim >= 8 * table_words {
            Some(self.vector(2 * table_words)?)
        } else { None };
        let words = [
            input.device_address(), output.device_address(),
            picks as u64, dim as u64,
            u64::from(gate_cap.to_bits()), u64::from(up_cap.to_bits()),
            u64::from(inv_gate.to_bits()), u64::from(inv_up.to_bits()),
            u64::from(vector), coefficients.as_ref().map_or(0, DeviceVector::device_address),
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        if coefficients.is_some() {
            let mut build = command;
            build.word(8, 2);
            let grid = [(table_words / fast_ops::CAPPED_GATE_THREADS) as u32, 1, 1, fast_ops::CAPPED_GATE_THREADS as u32, 0];
            if !self.record_resident_block(Kind::FastCappedGate, build, grid)? {
                return Err("fast capped gate: coefficient launch is not recorded now".into());
            }
        }
        let grid = [picks as u32, 1, 1, fast_ops::CAPPED_GATE_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastCappedGate, command, grid)? {
            return Err("fast capped gate: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Records the build (mode 2 of `FAST_CAPPED_GATE`) of the two exact coefficient tables of `CappedGate` into
    /// `tables` (`2 * CAPPED_GATE_TABLE_WORDS` private words): word `j` is the gate coefficient and word
    /// `CAPPED_GATE_TABLE_WORDS + j` the up coefficient of the BF16 value with bits `j`. `caps` = gate cap, up cap,
    /// reciprocal gate cap, reciprocal up cap. The same launch as the table build of `fast_capped_gate_into`; its
    /// input, output and shape words name the tables and one word, which mode 2 does not read.
    pub(crate) fn record_capped_gate_tables(&self, caps: [f32; 4], tables: &DeviceVector<'_>) -> Result<(), String> {
        let table_words = fast_ops::CAPPED_GATE_TABLE_WORDS;
        fast_ops::check_capped_gate_caps(caps[0], caps[1], caps[2], caps[3])?;
        self.check_vector(tables, 2 * table_words)?;
        if tables.is_segmented() || !tables.writable_from(0) { return Err("capped gate tables: private contiguous storage required".into()); }
        self.require_arm_kernels(&[Kind::FastCappedGate.kernel()])?;
        let at = tables.device_address();
        let mut command = Command::new(Kind::FastCappedGate);
        let words = [
            at, at, 1, 1,
            u64::from(caps[0].to_bits()), u64::from(caps[1].to_bits()),
            u64::from(caps[2].to_bits()), u64::from(caps[3].to_bits()),
            2, at,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let grid = [(table_words / fast_ops::CAPPED_GATE_THREADS) as u32, 1, 1, fast_ops::CAPPED_GATE_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastCappedGate, command, grid)? {
            return Err("capped gate tables: launch is not recorded now".into());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments, reason = "shape, flags and three device operands")]
    pub fn fast_router_pick_into(
        &self, rows: usize, experts: usize, picks: usize, normalize: bool, bound: f32,
        logits: &DeviceVector<'_>, bias: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.record_router_pick(rows, experts, picks, normalize, bound, logits, bias, output, None)
    }

    /// Opt-in `fast_router_pick` with single-block (`1 x 256`) LPT unique-expert compaction (`rows <= 8`).
    #[allow(clippy::too_many_arguments, reason = "shape, flags, operands and compacted dispatch output")]
    pub fn fast_router_pick_compact_into(
        &self, rows: usize, experts: usize, picks: usize, normalize: bool, bound: f32,
        logits: &DeviceVector<'_>, bias: &DeviceVector<'_>, output: &mut DeviceVector<'_>, compact: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.record_router_pick(rows, experts, picks, normalize, bound, logits, bias, output, Some(compact))
    }

    #[allow(clippy::too_many_arguments, reason = "shape, flags, operands and optional compacted output")]
    fn record_router_pick(
        &self, rows: usize, experts: usize, picks: usize, normalize: bool, bound: f32,
        logits: &DeviceVector<'_>, bias: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        compact: Option<&mut DeviceVector<'_>>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast router pick: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        if let Some(cmp) = compact.as_deref() {
            fast_ops::check_router_pick_compact(rows, experts, picks, bound, span(logits), span(bias), span(output), span(cmp))?;
            self.check_vector(cmp, rows * picks * fast_ops::ROUTER_PICK_COMPACT_ENTRY_WORDS)?;
            if cmp.is_segmented() || !cmp.writable_from(0) {
                return Err("fast router pick compact: compact buffer requires private contiguous storage".into());
            }
        } else {
            fast_ops::check_router_pick(rows, experts, picks, bound, span(logits), span(bias), span(output))?;
        }
        self.check_vector(logits, rows * experts)?;
        self.check_vector(bias, experts)?;
        self.check_vector(output, rows * 2 * picks)?;
        if logits.is_segmented() || bias.is_segmented() { return Err("fast router pick: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast router pick: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::FastRouterPick.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastRouterPick);
        let words = [
            logits.device_address(), bias.device_address(), output.device_address(),
            rows as u64, experts as u64, picks as u64, u64::from(normalize), u64::from(bound.to_bits()),
            compact.as_deref().map_or(0, DeviceVector::device_address),
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let grid = if compact.is_some() {
            [1, 1, 1, fast_ops::ROUTER_PICK_COMPACT_THREADS as u32, 0]
        } else {
            [rows.div_ceil(4) as u32, 1, 1, fast_ops::ROUTER_PICK_THREADS as u32, 0]
        };
        if !self.record_resident_block(Kind::FastRouterPick, command, grid)? {
            return Err("fast router pick: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    #[allow(clippy::too_many_arguments, reason = "shape, form and five device operands")]
    pub fn fast_short_conv_win_into(
        &self, rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
        window: &DeviceVector<'_>, token_taps: Option<&DeviceVector<'_>>, addend: Option<&DeviceVector<'_>>,
        fixed_taps: &DeviceVector<'_>, ring: Option<(&DeviceVector<'_>, usize)>, output: &mut DeviceVector<'_>,
        every: Option<(&mut DeviceVector<'_>, usize)>,
    ) -> Result<(), String> {
        self.record_short_conv(rows, heads, dim, taps, form, bound, window, token_taps, addend, fixed_taps, ring, output, every, None, None, None)
    }

    /// 2D tiled prefill `ShortConv` (`heads == 1`, no token taps, `taps in 2..=4`, `rows >= taps`)
    /// writing both `output` and optional `next_ring` (`[taps][width]` FP32) in one launch.
    #[allow(clippy::too_many_arguments, reason = "shape, form, device operands, and optional next ring")]
    pub fn fast_short_conv_win_ring_into(
        &self, rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
        window: &DeviceVector<'_>, token_taps: Option<&DeviceVector<'_>>, addend: Option<&DeviceVector<'_>>,
        fixed_taps: &DeviceVector<'_>, ring: Option<(&DeviceVector<'_>, usize)>, output: &mut DeviceVector<'_>,
        next_ring: Option<&mut DeviceVector<'_>>,
    ) -> Result<(), String> {
        self.record_short_conv(rows, heads, dim, taps, form, bound, window, token_taps, addend, fixed_taps, ring, output, None, None, None, next_ring)
    }

    #[allow(clippy::too_many_arguments, reason = "shape, multi-chat descriptor and device operands")]
    pub fn fast_short_conv_multi_into(
        &self, rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
        window: &DeviceVector<'_>, token_taps: Option<&DeviceVector<'_>>, addend: Option<&DeviceVector<'_>>,
        fixed_taps: &DeviceVector<'_>, ring: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        every: &mut DeviceVector<'_>, multi: fast_ops::ShortConvMulti,
    ) -> Result<(), String> {
        self.record_short_conv(rows, heads, dim, taps, form, bound, window, token_taps, addend, fixed_taps, Some((ring, 0)), output, Some((every, 0)), None, Some(multi), None)
    }

    /// Direct per-chat ring pointer form of `fast_short_conv_multi_into`: passes up to `SHORT_CONV_MULTI_CHATS_MAX`
    /// (64) chat ring device pointers in payload words `20..20 + chats`, eliminating `compute.join` ring copies.
    #[allow(clippy::too_many_arguments, reason = "shape, multi-chat descriptor, direct per-chat rings and device operands")]
    pub fn fast_short_conv_multi_rings_into(
        &self, rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
        window: &DeviceVector<'_>, token_taps: Option<&DeviceVector<'_>>, addend: Option<&DeviceVector<'_>>,
        fixed_taps: &DeviceVector<'_>, rings: &[&DeviceVector<'_>], output: &mut DeviceVector<'_>,
        every: &mut DeviceVector<'_>, multi: fast_ops::ShortConvMulti,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast short conv: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let ring_spans: Vec<Span> = rings.iter().map(|&v| span(v)).collect();
        fast_ops::check_short_conv_multi_rings(
            rows, heads, dim, taps, form, bound, span(window), token_taps.map(span), addend.map(span),
            span(fixed_taps), &ring_spans, span(output), span(every), multi,
        )?;
        let width = heads * dim;
        self.check_vector(window, rows * width)?;
        if let Some(tt) = token_taps { self.check_vector(tt, rows * heads * taps)?; if tt.is_segmented() { return Err("fast short conv: token_taps segmented".into()); } }
        if let Some(ad) = addend { self.check_vector(ad, rows * width)?; if ad.is_segmented() { return Err("fast short conv: addend segmented".into()); } }
        self.check_vector(fixed_taps, width * taps)?;
        self.check_vector(output, rows * width)?;
        self.check_vector(every, rows * taps * width)?;
        if window.is_segmented() || fixed_taps.is_segmented() { return Err("fast short conv: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast short conv: output requires private contiguous storage".into()); }
        if every.is_segmented() || !every.writable_from(0) { return Err("fast short conv: kept rings require private contiguous storage".into()); }
        for &r in rings {
            self.check_vector(r, taps * width)?;
            if r.is_segmented() { return Err("fast short conv: ring segmented".into()); }
        }
        self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastStreamOps);
        let tag = (fast_ops::SHORT_CONV_MULTI_TAG | multi.chats) as u64;
        let words = [
            fast_ops::STREAM_OP_SHORT_CONV_WIN,
            window.device_address(),
            token_taps.map_or(0, DeviceVector::device_address),
            addend.map_or(0, DeviceVector::device_address),
            fixed_taps.device_address(),
            output.device_address(),
            rows as u64, width as u64, taps as u64, heads as u64, u64::from(form), u64::from(bound.to_bits()),
            0, tag,
            every.device_address(), tag,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        command.word(fast_ops::SHORT_CONV_MULTI_MASK_0, multi.start_mask[0]);
        command.word(fast_ops::SHORT_CONV_MULTI_MASK_1, multi.start_mask[1]);
        command.word(fast_ops::SHORT_CONV_MULTI_SLOTS_LO, multi.slots_lo);
        command.word(fast_ops::SHORT_CONV_MULTI_SLOTS_HI, multi.slots_hi);
        command.word(fast_ops::SHORT_CONV_MULTI_MASK_2, multi.start_mask[2]);
        command.word(fast_ops::SHORT_CONV_MULTI_MASK_3, multi.start_mask[3]);
        for (c, &r) in rings.iter().enumerate() {
            command.word(fast_ops::SHORT_CONV_MULTI_RINGS_AT + c, r.device_address());
        }
        let blocks = fast_ops::stream_ops_blocks(rows, width);
        let grid = [u32::try_from(blocks).map_err(|_| "fast short conv: grid overflow")?, 1, 1, fast_ops::STREAM_OPS_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastStreamOps, command, grid)? {
            return Err("fast short conv: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Fused `Softcap` (`cap(rn(x), eps, bound)`) + greedy argmax (`[rows][3]`: `best, id, finite`),
    /// eliminating the `[rows][cols]` capped-logits write and reload on greedy decode & verify steps.
    #[allow(clippy::too_many_arguments, reason = "shape, softcap parameters and device operands")]
    pub fn fast_softcap_argmax_into(
        &self, rows: usize, cols: usize, bound: f32, eps: f32,
        input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast softcap argmax: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let parts = fast_ops::softcap_argmax_parts(rows, cols);
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        self.check_vector(input, rows * cols)?;
        self.check_vector(output, rows * 3)?;
        if input.is_segmented() { return Err("fast softcap argmax: input must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) {
            return Err("fast softcap argmax: output requires private contiguous storage".into());
        }
        self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
        self.resident_completion()?;
        let make_cmd = |src: u64, dst: u64, finish: bool| {
            let mut command = Command::new(Kind::FastStreamOps);
            let words = [
                fast_ops::STREAM_OP_SOFTCAP_ARGMAX,
                src, 0, 0, 0, dst,
                rows as u64, cols as u64, parts as u64, u64::from(finish),
                u64::from(eps.to_bits()), u64::from(bound.to_bits()),
            ];
            for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
            command
        };
        if parts == 1 {
            fast_ops::check_softcap_argmax(rows, cols, 1, false, bound, eps, span(input), span(output))?;
            let command = make_cmd(input.device_address(), output.device_address(), false);
            let grid = [u32::try_from(rows).map_err(|_| "fast softcap argmax: grid overflow")?, 1, 1, fast_ops::STREAM_OPS_THREADS as u32, 0];
            if !self.record_resident_block(Kind::FastStreamOps, command, grid)? {
                return Err("fast softcap argmax: launches are not recorded now".into());
            }
            return self.resident_completion();
        }
        let partials = self.vector(rows * parts * 3)?;
        fast_ops::check_softcap_argmax(rows, cols, parts, false, bound, eps, span(input), span(&partials))?;
        fast_ops::check_softcap_argmax(rows, cols, parts, true, bound, eps, span(&partials), span(output))?;
        let scan_cmd = make_cmd(input.device_address(), partials.device_address(), false);
        let fin_cmd = make_cmd(partials.device_address(), output.device_address(), true);
        let r_u32 = u32::try_from(rows).map_err(|_| "fast softcap argmax: grid overflow")?;
        let p_u32 = u32::try_from(parts).map_err(|_| "fast softcap argmax: grid overflow")?;
        let scan_grid = [r_u32, p_u32, 1, fast_ops::STREAM_OPS_THREADS as u32, 0];
        let fin_grid = [r_u32, 1, 1, fast_ops::STREAM_OPS_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastStreamOps, scan_cmd, scan_grid)?
            || !self.record_resident_block(Kind::FastStreamOps, fin_cmd, fin_grid)?
        {
            return Err("fast softcap argmax: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Convolve prompt keys or values, finish each head and store its packed words directly in the KV plane.
    #[allow(clippy::too_many_arguments, reason = "shape, taps, and KV plane")]
    pub fn fast_short_conv_kv_into(
        &self, shape: crate::block_ops::Shape, rows: usize, window: &DeviceVector<'_>, token_taps: Option<&DeviceVector<'_>>,
        fixed_taps: &DeviceVector<'_>, ring: (&DeviceVector<'_>, usize), plane: &mut DeviceVector<'_>, kv: fast_ops::KvWrite,
    ) -> Result<(), String> {
        self.record_short_conv(rows, shape.heads as usize, shape.dim as usize, shape.taps as usize, shape.cols, shape.bound,
            window, token_taps, None, fixed_taps, Some(ring), plane, None, Some(kv), None, None)
    }

    /// Convolve prompt keys or values, finish each head into the KV plane, and write `next_ring` in the same launch.
    #[allow(clippy::too_many_arguments, reason = "shape, taps, KV plane, and optional next ring")]
    pub fn fast_short_conv_kv_ring_into(
        &self, shape: crate::block_ops::Shape, rows: usize, window: &DeviceVector<'_>, token_taps: Option<&DeviceVector<'_>>,
        fixed_taps: &DeviceVector<'_>, ring: (&DeviceVector<'_>, usize), plane: &mut DeviceVector<'_>, kv: fast_ops::KvWrite,
        next_ring: Option<&mut DeviceVector<'_>>,
    ) -> Result<(), String> {
        self.record_short_conv(rows, shape.heads as usize, shape.dim as usize, shape.taps as usize, shape.cols, shape.bound,
            window, token_taps, None, fixed_taps, Some(ring), plane, None, Some(kv), None, next_ring)
    }

    #[allow(clippy::too_many_arguments, reason = "shape, form and device operands")]
    fn record_short_conv(
        &self, rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
        window: &DeviceVector<'_>, token_taps: Option<&DeviceVector<'_>>, addend: Option<&DeviceVector<'_>>,
        fixed_taps: &DeviceVector<'_>, ring: Option<(&DeviceVector<'_>, usize)>, output: &mut DeviceVector<'_>,
        every: Option<(&mut DeviceVector<'_>, usize)>, kv: Option<fast_ops::KvWrite>, multi: Option<fast_ops::ShortConvMulti>,
        next_ring: Option<&mut DeviceVector<'_>>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast short conv: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let width = heads * dim;
        let out_span = kv.map_or_else(
            || span(output),
            |kv| Span { address: output.device_address(), bytes: (kv.plane_rows * (width / 2)) as u64 * 4 },
        );
        fast_ops::check_short_conv_form(rows, heads, dim, taps, form, bound, span(window), token_taps.map(span), addend.map(span),
            span(fixed_taps), ring.map(|(v, first)| (span(v), first)), out_span, every.as_ref().map(|(v, first)| (span(v), *first)), kv, multi)?;
        let history = if ring.is_some() { 0 } else { taps - 1 };
        self.check_vector(window, (rows + history) * width)?;
        if let Some((v, _)) = ring {
            let chats = multi.map_or(1, |m| m.chats);
            self.check_vector(v, chats * taps * width)?;
            if v.is_segmented() { return Err("fast short conv: ring segmented".into()); }
        }
        if let Some(tt) = token_taps { self.check_vector(tt, rows * heads * taps)?; if tt.is_segmented() { return Err("fast short conv: token_taps segmented".into()); } }
        if let Some(ad) = addend { self.check_vector(ad, rows * width)?; if ad.is_segmented() { return Err("fast short conv: addend segmented".into()); } }
        self.check_vector(fixed_taps, width * taps)?;
        if window.is_segmented() || fixed_taps.is_segmented() { return Err("fast short conv: operand must be contiguous".into()); }
        if let Some(kv) = kv {
            self.check_vector_prefix(output, kv.plane_rows * (width / 2))?;
            let write_start = if kv.first + rows <= kv.plane_rows { kv.first * (width / 2) } else { 0 };
            if !output.writable_from(write_start) {
                return Err("fast short conv: KV plane is shared".into());
            }
            let full_out = span(output);
            let reads = [Some(span(window)), token_taps.map(span), addend.map(span), Some(span(fixed_taps)), ring.map(|(v, _)| span(v)), every.as_ref().map(|(v, _)| span(v))];
            if reads.iter().flatten().any(|r| full_out.meets(*r)) {
                return Err("fast short conv: KV plane overlaps operand".into());
            }
        } else {
            self.check_vector(output, rows * width)?;
            if output.is_segmented() || !output.writable_from(0) {
                return Err("fast short conv: output requires private contiguous storage".into());
            }
        }
        if let Some((v, _)) = &every {
            self.check_vector(v, rows * taps * width)?;
            if v.is_segmented() || !v.writable_from(0) { return Err("fast short conv: kept rings require private contiguous storage".into()); }
        }
        let tiled = kv.is_none() && ring.is_some() && every.is_none()
            && heads == 1 && (form & crate::block_stream::CONV_TOKEN_TAPS) == 0
            && (2..=fast_ops::ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) && rows >= 2;
        let kv_tiled = kv.is_some() && (2..=fast_ops::ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) && rows >= taps;
        if let Some(nr) = next_ring.as_deref() {
            if !(kv_tiled || (tiled && rows >= taps)) {
                return Err("fast short conv: next_ring requires tiled prefill launch".into());
            }
            self.check_vector(nr, taps * width)?;
            if nr.is_segmented() || !nr.writable_from(0) {
                return Err("fast short conv: next_ring requires private contiguous storage".into());
            }
            let nr_span = span(nr);
            let other = [Some(span(window)), token_taps.map(span), addend.map(span), Some(span(fixed_taps)), ring.map(|(v, _)| span(v)), Some(span(output))];
            if other.iter().flatten().any(|s| nr_span.meets(*s)) {
                return Err("fast short conv: next_ring overlaps operand".into());
            }
        }
        let kind = if kv.is_some() { Kind::FastRowDots } else { Kind::FastStreamOps };
        self.require_arm_kernels(&[kind.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(kind);
        let ring_first = multi.map_or_else(|| ring.map_or(0, |(_, first)| first as u64), |m| (fast_ops::SHORT_CONV_MULTI_TAG | m.chats) as u64);
        let every_first = multi.map_or_else(|| every.as_ref().map_or(0, |(_, first)| *first as u64), |m| (fast_ops::SHORT_CONV_MULTI_TAG | m.chats) as u64);
        let words = [
            fast_ops::STREAM_OP_SHORT_CONV_WIN,
            window.device_address(),
            token_taps.map_or(0, DeviceVector::device_address),
            addend.map_or(0, DeviceVector::device_address),
            fixed_taps.device_address(),
            output.device_address(),
            rows as u64, width as u64, taps as u64, heads as u64, u64::from(form), u64::from(bound.to_bits()),
            ring.map_or(0, |(v, _)| v.device_address()), ring_first,
            every.as_ref().map_or(0, |(v, _)| v.device_address()), every_first,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        if let Some(m) = multi {
            command.word(fast_ops::SHORT_CONV_MULTI_MASK_0, m.start_mask[0]);
            command.word(fast_ops::SHORT_CONV_MULTI_MASK_1, m.start_mask[1]);
            command.word(fast_ops::SHORT_CONV_MULTI_SLOTS_LO, m.slots_lo);
            command.word(fast_ops::SHORT_CONV_MULTI_SLOTS_HI, m.slots_hi);
            command.word(fast_ops::SHORT_CONV_MULTI_MASK_2, m.start_mask[2]);
            command.word(fast_ops::SHORT_CONV_MULTI_MASK_3, m.start_mask[3]);
        }
        let threads = if kv.is_some() { fast_ops::ROW_DOTS_THREADS } else { fast_ops::STREAM_OPS_THREADS };
        let grid = if tiled {
            let tile = fast_ops::SHORT_CONV_PREFILL_TILE;
            command.word(15, tile as u64);
            command.word(fast_ops::SHORT_CONV_NEXT_RING_WORD, next_ring.as_ref().map_or(0, |v| v.device_address()));
            let gx = u32::try_from(width.div_ceil(threads)).map_err(|_| "fast short conv: grid overflow")?;
            let gy = u32::try_from(rows.div_ceil(tile)).map_err(|_| "fast short conv: grid overflow")?;
            [gx, gy, 1, threads as u32, 0]
        } else if let Some(kv) = kv {
            command.word(fast_ops::KV_FINISH_WORD, if kv.clip { fast_ops::CONV_KV_CLIP } else { fast_ops::CONV_KV_NORM });
            command.word(17, u64::from(kv.param.to_bits()));
            command.word(18, kv.first as u64);
            command.word(19, kv.plane_rows as u64);
            if kv_tiled {
                let tile = fast_ops::kv_prefill_tile(rows);
                command.word(fast_ops::KV_NEXT_RING_WORD, next_ring.as_ref().map_or(0, |v| v.device_address()));
                command.word(fast_ops::KV_PREFILL_TILE_WORD, tile as u64);
                let gx = u32::try_from(heads).map_err(|_| "fast short conv: grid overflow")?;
                let gy = u32::try_from(fast_ops::conv_kv_tiled_blocks_y(rows, tile)).map_err(|_| "fast short conv: grid overflow")?;
                [gx, gy, 1, threads as u32, 0]
            } else {
                let blocks = fast_ops::conv_kv_blocks(rows, heads);
                [u32::try_from(blocks).map_err(|_| "fast short conv: grid overflow")?, 1, 1, threads as u32, 0]
            }
        } else {
            let blocks = if ring.is_some() && every.is_none() {
                rows
            } else {
                fast_ops::stream_ops_blocks(rows, width)
            };
            [u32::try_from(blocks).map_err(|_| "fast short conv: grid overflow")?, 1, 1, threads as u32, 0]
        };
        if !self.record_resident_block(kind, command, grid)? {
            return Err("fast short conv: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// The tiled prefill convolution over the half form of its input (`fast_ops::SHORT_CONV_HALF_IN_WORD`):
    /// `window` holds `rows * width` BF16 values, the output of `project_tc_big_half_out_into`, which also wrote
    /// the ring after the step. The words of `output` are those of `fast_short_conv_win_ring_into` over the product
    /// words that `window` is the rounding of.
    #[allow(clippy::too_many_arguments, reason = "shape, form and device operands")]
    pub fn fast_short_conv_half_into(
        &self, rows: usize, width: usize, taps: usize, form: u32, bound: f32,
        window: &DeviceVector<'_>, addend: Option<&DeviceVector<'_>>, fixed_taps: &DeviceVector<'_>,
        ring: (&DeviceVector<'_>, usize), output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast short conv half: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        fast_ops::check_short_conv_half(rows, width, taps, form, bound, span(window), addend.map(span), span(fixed_taps), (span(ring.0), ring.1), span(output))?;
        self.check_vector(window, rows * width / 2)?;
        self.check_vector(fixed_taps, width * taps)?;
        self.check_vector(ring.0, taps * width)?;
        self.check_vector(output, rows * width)?;
        if let Some(ad) = addend { self.check_vector(ad, rows * width)?; if ad.is_segmented() { return Err("fast short conv half: addend segmented".into()); } }
        if window.is_segmented() || fixed_taps.is_segmented() || ring.0.is_segmented() { return Err("fast short conv half: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast short conv half: output requires private contiguous storage".into()); }
        self.skip_fill(output)?; // the launch stores every word: one thread a channel, every row of its tile
        self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastStreamOps);
        let tile = fast_ops::SHORT_CONV_PREFILL_TILE;
        let words = [
            fast_ops::STREAM_OP_SHORT_CONV_WIN, window.device_address(), 0, addend.map_or(0, DeviceVector::device_address),
            fixed_taps.device_address(), output.device_address(),
            rows as u64, width as u64, taps as u64, 1, u64::from(form), u64::from(bound.to_bits()),
            ring.0.device_address(), ring.1 as u64, 0, tile as u64,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        command.word(fast_ops::SHORT_CONV_NEXT_RING_WORD, 0);
        command.word(fast_ops::SHORT_CONV_HALF_IN_WORD, 1);
        let threads = fast_ops::STREAM_OPS_THREADS;
        let gx = u32::try_from(width.div_ceil(threads)).map_err(|_| "fast short conv half: grid overflow")?;
        let gy = u32::try_from(rows.div_ceil(tile)).map_err(|_| "fast short conv half: grid overflow")?;
        if !self.record_resident_block(Kind::FastStreamOps, command, [gx, gy, 1, threads as u32, 0])? {
            return Err("fast short conv half: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Multi-row paired KV finish (`Kind::FastRowDots` with `KV_FINISH_TAG`): runs Key (`ShortConv` + `HeadClip` +
    /// `PackBf16` + optional `every`) and Value (`ShortConv` + `RoundedNorm` + `PackBf16` + optional `every`) for
    /// `head_dim == 128` in one 128-thread launch.
    #[allow(clippy::too_many_arguments, reason = "paired key/value shapes, scalars, and device operands")]
    pub fn fast_kv_finish_pair_into(
        &self, rows: usize, kv_heads: usize, taps: usize, first_slot: usize,
        key_bound: f32, val_eps: f32, key_scale: f32, val_scale: f32,
        key_in: &DeviceVector<'_>, key_ring: &DeviceVector<'_>, key_fixed: &DeviceVector<'_>,
        key_made: Option<&DeviceVector<'_>>, key_packed: &mut DeviceVector<'_>, key_every: Option<&mut DeviceVector<'_>>,
        val_in: &DeviceVector<'_>, val_ring: &DeviceVector<'_>, val_fixed: &DeviceVector<'_>,
        val_made: Option<&DeviceVector<'_>>, val_packed: &mut DeviceVector<'_>, val_every: Option<&mut DeviceVector<'_>>,
        chats: &[(&DeviceVector<'_>, &DeviceVector<'_>, usize)],
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast kv finish: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let width = kv_heads.checked_mul(fast_ops::KV_FINISH_HEAD_DIM).ok_or("fast kv finish: width overflow")?;
        let check = |k: &DeviceVector<'_>, v: &DeviceVector<'_>, slot| {
            fast_ops::check_kv_finish_pair(
                rows, kv_heads, taps, slot, key_bound, val_eps, key_scale, val_scale,
                span(key_in), span(k), span(key_fixed), key_made.map(span), span(key_packed), key_every.as_deref().map(span),
                span(val_in), span(v), span(val_fixed), val_made.map(span), span(val_packed), val_every.as_deref().map(span),
            )
        };
        check(key_ring, val_ring, first_slot)?;
        if !chats.is_empty() && (chats.len() != rows || rows > 16) { return Err("fast kv finish: chat ring count differs".into()); }
        for &(k, v, slot) in chats {
            check(k, v, slot)?;
            for ring in [k, v] {
                self.check_vector(ring, taps * width)?;
                if ring.is_segmented() { return Err("fast kv finish: segmented chat ring".into()); }
            }
        }
        for (inp, rng, fix, mad, pck, ev) in [
            (key_in, key_ring, key_fixed, key_made, &*key_packed, key_every.as_deref()),
            (val_in, val_ring, val_fixed, val_made, &*val_packed, val_every.as_deref()),
        ] {
            self.check_vector(inp, rows * width)?;
            self.check_vector(rng, taps * width)?;
            self.check_vector(fix, width * taps)?;
            self.check_vector(pck, rows * width / 2)?;
            if inp.is_segmented() || rng.is_segmented() || fix.is_segmented() { return Err("fast kv finish: input segmented".into()); }
            if let Some(m) = mad { self.check_vector(m, rows * kv_heads * taps)?; if m.is_segmented() { return Err("fast kv finish: made segmented".into()); } }
            if pck.is_segmented() || !pck.writable_from(0) { return Err("fast kv finish: packed output requires private contiguous storage".into()); }
            if let Some(e) = ev {
                self.check_vector(e, rows * taps * width)?;
                if e.is_segmented() || !e.writable_from(0) { return Err("fast kv finish: every output requires private contiguous storage".into()); }
            }
        }
        self.require_arm_kernels(&[Kind::FastRowDots.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastRowDots);
        let words = [
            fast_ops::KV_FINISH_TAG, 0,
            rows as u64, kv_heads as u64, taps as u64, first_slot as u64,
            u64::from(key_bound.to_bits()) | (u64::from(val_eps.to_bits()) << 32),
            u64::from(key_scale.to_bits()) | (u64::from(val_scale.to_bits()) << 32),
            key_in.device_address(), key_ring.device_address(), key_fixed.device_address(),
            key_made.map_or(0, DeviceVector::device_address), key_packed.device_address(),
            key_every.map_or(0, |v| v.device_address()),
            val_in.device_address(), val_ring.device_address(), val_fixed.device_address(),
            val_made.map_or(0, DeviceVector::device_address), val_packed.device_address(),
            val_every.map_or(0, |v| v.device_address()),
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        command.word(20, chats.len() as u64);
        for (row, &(k, v, slot)) in chats.iter().enumerate() {
            command.word(21 + 3 * row, k.device_address());
            command.word(22 + 3 * row, v.device_address());
            command.word(23 + 3 * row, slot as u64);
        }
        let blocks = u32::try_from((2 * rows * kv_heads).div_ceil(4)).map_err(|_| "fast kv finish: grid overflow")?;
        let grid = [blocks, 1, 1, fast_ops::KV_FINISH_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastRowDots, command, grid)? {
            return Err("fast kv finish: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Fused multi-row `RouteFold` + `AfterExperts` `ShortConv` + `every` + `Residual` + `Round`
    /// (`STREAM_OP_ROUTE_FOLD_CONV`), eliminating both the `RouteFold` launch and its intermediate buffer.
    #[allow(clippy::too_many_arguments, reason = "shape, strided route layout, ring slot, and device operands")]
    pub fn fast_route_fold_conv_into(
        &self, rows: usize, picks: usize, dim: usize, weight_stride: usize, weight_offset: usize,
        taps: usize, first_slot: usize, bound: f32,
        route: &DeviceVector<'_>, downs: &DeviceVector<'_>, residual: &DeviceVector<'_>,
        fixed: &DeviceVector<'_>, ring: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        every: Option<&mut DeviceVector<'_>>, chats: &[(&DeviceVector<'_>, usize)],
    ) -> Result<(), String> {
        self.fast_route_fold_conv_form_into(rows, picks, dim, weight_stride, weight_offset, taps, first_slot, bound,
            route, downs, residual, fixed, ring, output, every, chats, false)
    }

    #[allow(clippy::too_many_arguments, reason = "shape, strided route layout, ring slot, device operands and down form")]
    pub fn fast_route_fold_conv_form_into(
        &self, rows: usize, picks: usize, dim: usize, weight_stride: usize, weight_offset: usize,
        taps: usize, first_slot: usize, bound: f32,
        route: &DeviceVector<'_>, downs: &DeviceVector<'_>, residual: &DeviceVector<'_>,
        fixed: &DeviceVector<'_>, ring: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        every: Option<&mut DeviceVector<'_>>, chats: &[(&DeviceVector<'_>, usize)], half: bool,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast route fold conv: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let check_ring = |r: &DeviceVector<'_>, slot| fast_ops::check_route_fold_conv_form(
            rows, picks, dim, weight_stride, weight_offset, taps, slot, bound,
            span(route), span(downs), span(residual), span(fixed), span(r), span(output), every.as_deref().map(span), half,
        );
        check_ring(ring, first_slot)?;
        if !chats.is_empty() && (chats.len()!=rows || rows>16) { return Err("route fold: chat count differs".into()); }
        for &(r, slot) in chats {
            check_ring(r, slot)?;
            self.check_vector(r, taps*dim)?;
            if r.is_segmented() { return Err("route fold: segmented chat ring".into()); }
        }
        self.check_vector(route, rows * weight_stride)?;
        self.check_vector(downs, if half { rows * picks * dim / 2 } else { rows * picks * dim })?;
        self.check_vector(residual, rows * dim)?;
        self.check_vector(fixed, dim * taps)?;
        self.check_vector(ring, taps * dim)?;
        self.check_vector(output, rows * dim)?;
        if route.is_segmented() || downs.is_segmented() || residual.is_segmented() || fixed.is_segmented() || ring.is_segmented() {
            return Err("fast route fold conv: operand must be contiguous".into());
        }
        if output.is_segmented() || !output.writable_from(0) {
            return Err("fast route fold conv: output requires private contiguous storage".into());
        }
        if let Some(ev) = every.as_deref() {
            self.check_vector(ev, rows * taps * dim)?;
            if ev.is_segmented() || !ev.writable_from(0) {
                return Err("fast route fold conv: every requires private contiguous storage".into());
            }
        }
        self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastStreamOps);
        let words = [
            fast_ops::STREAM_OP_ROUTE_FOLD_CONV,
            route.device_address(), downs.device_address(), residual.device_address(),
            fixed.device_address(), output.device_address(),
            rows as u64, dim as u64, picks as u64, weight_stride as u64, weight_offset as u64,
            u64::from(bound.to_bits()), ring.device_address(), taps as u64,
            every.map_or(0, |v| v.device_address()), first_slot as u64,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        command.word(16, chats.len() as u64);
        for (row, &(r, slot)) in chats.iter().enumerate() {
            command.word(17 + 2 * row, r.device_address());
            command.word(18 + 2 * row, slot as u64);
        }
        command.word(fast_ops::ROUTE_FOLD_CONV_FORM_WORD, if half { fast_ops::FOLD_HALF } else { 0 });
        let blocks = u32::try_from(dim.div_ceil(fast_ops::ROUTE_FOLD_CONV_THREADS)).map_err(|_| "fast route fold conv: grid overflow")?;
        let grid = [blocks, chats.len().max(1) as u32, 1, fast_ops::ROUTE_FOLD_CONV_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastStreamOps, command, grid)? {
            return Err("fast route fold conv: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Prefill fused `RouteFold (FOLD_HALF)` + `AfterExperts ShortConv` + `next_ring` (`STREAM_OP_ROUTE_FOLD_CONV_PREFILL`).
    #[allow(clippy::too_many_arguments, reason = "shape, ring parameters, and eight device operands")]
    pub fn fast_route_fold_conv_prefill_into(
        &self, rows: usize, picks: usize, dim: usize, taps: usize, first_slot: usize, bound: f32,
        weights: &DeviceVector<'_>, downs_half: &DeviceVector<'_>, residual: &DeviceVector<'_>,
        fixed: &DeviceVector<'_>, ring: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        next_ring: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast route fold conv prefill: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        fast_ops::check_route_fold_conv_prefill(
            rows, picks, dim, taps, first_slot, bound,
            span(weights), span(downs_half), span(residual), span(fixed), span(ring), span(output), span(next_ring),
        )?;
        self.check_vector(weights, rows * picks)?;
        self.check_vector(downs_half, rows * picks * dim / 2)?;
        self.check_vector(residual, rows * dim)?;
        self.check_vector(fixed, dim * taps)?;
        self.check_vector(ring, taps * dim)?;
        self.check_vector(output, rows * dim)?;
        self.check_vector(next_ring, taps * dim)?;
        if [weights, downs_half, residual, fixed, ring].iter().any(|v| v.is_segmented()) {
            return Err("fast route fold conv prefill: input segmented".into());
        }
        if output.is_segmented() || !output.writable_from(0) || next_ring.is_segmented() || !next_ring.writable_from(0) {
            return Err("fast route fold conv prefill: output requires private contiguous storage".into());
        }
        self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastStreamOps);
        let tile = fast_ops::route_fold_prefill_tile(rows);
        let words = [
            fast_ops::STREAM_OP_ROUTE_FOLD_CONV_PREFILL,
            weights.device_address(), downs_half.device_address(), residual.device_address(),
            fixed.device_address(), output.device_address(),
            rows as u64, dim as u64, picks as u64, taps as u64, first_slot as u64, u64::from(bound.to_bits()),
            ring.device_address(), next_ring.device_address(), tile as u64,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let gx = u32::try_from((dim / 2).div_ceil(fast_ops::ROUTE_FOLD_PREFILL_THREADS)).map_err(|_| "fast route fold conv prefill: grid overflow")?;
        let gy = u32::try_from(rows.div_ceil(tile)).map_err(|_| "fast route fold conv prefill: grid overflow")?;
        let grid = [gx, gy, 1, fast_ops::ROUTE_FOLD_PREFILL_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastStreamOps, command, grid)? {
            return Err("fast route fold conv prefill: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    pub fn fast_route_fold_into(
        &self, rows: usize, picks: usize, dim: usize,
        weights: &DeviceVector<'_>, downs: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.route_fold_into(rows, picks, dim, picks, 0, weights, downs, output, false)
    }

    #[allow(clippy::too_many_arguments, reason = "shape, strided weight layout and operands")]
    pub fn fast_route_fold_strided_into(
        &self, rows: usize, picks: usize, dim: usize, weight_stride: usize, weight_offset: usize,
        weights: &DeviceVector<'_>, downs: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.route_fold_into(rows, picks, dim, weight_stride, weight_offset, weights, downs, output, false)
    }

    /// `fast_route_fold_into` on half downs (`fast_ops::FOLD_HALF`): `downs` holds `rows x picks x dim` BF16 values,
    /// two a word (`project_tc_packed_down_half_into`), `dim` even. The words equal the FP32 form on the FP32 downs.
    pub fn fast_route_fold_half_into(
        &self, rows: usize, picks: usize, dim: usize,
        weights: &DeviceVector<'_>, downs: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        self.route_fold_into(rows, picks, dim, picks, 0, weights, downs, output, true)
    }

    #[allow(clippy::too_many_arguments, reason = "shape, strided weight layout, operands and down form")]
    fn route_fold_into(
        &self, rows: usize, picks: usize, dim: usize, weight_stride: usize, weight_offset: usize,
        weights: &DeviceVector<'_>, downs: &DeviceVector<'_>, output: &mut DeviceVector<'_>, half: bool,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast route fold: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        fast_ops::check_route_fold_strided(rows, picks, dim, weight_stride, weight_offset, span(weights), span(downs), span(output), half)?;
        self.check_vector(weights, rows * weight_stride)?;
        self.check_vector(downs, if half { rows * picks * dim / 2 } else { rows * picks * dim })?;
        self.check_vector(output, rows * dim)?;
        if weights.is_segmented() || downs.is_segmented() { return Err("fast route fold: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast route fold: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastStreamOps);
        let words = [
            fast_ops::STREAM_OP_ROUTE_FOLD,
            weights.device_address(), downs.device_address(), 0, 0, output.device_address(),
            rows as u64, dim as u64, picks as u64, weight_stride as u64, weight_offset as u64, 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        command.word(fast_ops::FOLD_FORM_WORD, if half { fast_ops::FOLD_HALF } else { 0 });
        let grid = [u32::try_from(fast_ops::stream_ops_blocks(rows, dim)).map_err(|_| "fast route fold: grid overflow")?, 1, 1, fast_ops::STREAM_OPS_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastStreamOps, command, grid)? {
            return Err("fast route fold: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    pub fn fast_gate_blend_into(
        &self, rows: usize, dim: usize,
        gates: &DeviceVector<'_>, kept: &DeviceVector<'_>, msg: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast gate blend: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        fast_ops::check_gate_blend(rows, dim, span(gates), span(kept), span(msg), span(output))?;
        self.check_vector(gates, rows * 2 * dim)?;
        self.check_vector(kept, rows * dim)?;
        self.check_vector(msg, rows * dim)?;
        self.check_vector(output, rows * dim)?;
        if gates.is_segmented() || kept.is_segmented() || msg.is_segmented() { return Err("fast gate blend: operand must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("fast gate blend: output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastStreamOps);
        let words = [
            fast_ops::STREAM_OP_GATE_BLEND,
            gates.device_address(), kept.device_address(), msg.device_address(), 0, output.device_address(),
            rows as u64, dim as u64, 0, 0, 0, 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let grid = [u32::try_from(fast_ops::stream_ops_blocks(rows, dim)).map_err(|_| "fast gate blend: grid overflow")?, 1, 1, fast_ops::STREAM_OPS_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastStreamOps, command, grid)? {
            return Err("fast gate blend: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Fused `GateBlend` + optional weighted `RoundedNorm` for 1..=3 gates in one `FAST_STREAM_OPS` launch.
    pub fn fast_gate_blend_norm(
        &self, rows: usize, dim: usize, eps: f32,
        entries: &[(&DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>, Option<&DeviceVector<'_>>)],
    ) -> Result<Vec<DeviceVector<'a>>, String> {
        if self.arm_capture() { return Err("fast gate blend norm: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        if !self.resident_launches_active() { return Err("fast gate blend norm: launches are not recorded now".into()); }
        let count = entries.len();
        if count == 0 || count > fast_ops::GATE_BLEND_NORM_GATES_MAX {
            return Err("fast gate blend norm: gate count outside envelope".into());
        }
        let g_words = rows.checked_mul(2).and_then(|n| n.checked_mul(dim)).ok_or("fast gate blend norm: gate overflow")?;
        let v_words = rows.checked_mul(dim).ok_or("fast gate blend norm: row overflow")?;
        let mut outputs = Vec::with_capacity(count);
        for &(gates, kept, msg, weight) in entries {
            self.check_vector(gates, g_words)?;
            self.check_vector(kept, v_words)?;
            self.check_vector(msg, v_words)?;
            if gates.is_segmented() || kept.is_segmented() || msg.is_segmented() {
                return Err("fast gate blend norm: operand must be contiguous".into());
            }
            if let Some(wt) = weight {
                self.check_vector(wt, dim)?;
                if wt.is_segmented() { return Err("fast gate blend norm: weight must be contiguous".into()); }
            }
            outputs.push(self.vector(v_words)?);
        }
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let spans: Vec<_> = entries.iter().zip(&outputs).map(|(&(g, k, m, wt), o)| (span(g), span(k), span(m), wt.map(span), span(o))).collect();
        fast_ops::check_gate_blend_norm(rows, dim, eps, &spans)?;
        self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastStreamOps);
        for (i, w) in [
            fast_ops::STREAM_OP_GATE_BLEND_NORM,
            0, 0, 0, 0, 0,
            rows as u64, dim as u64, count as u64, 0, 0, u64::from(eps.to_bits()),
        ].into_iter().enumerate() {
            command.word(i, w);
        }
        for (idx, (&(g, k, m, wt), o)) in entries.iter().zip(&outputs).enumerate() {
            let base = fast_ops::GATE_BLEND_NORM_TABLE_AT + idx * 5;
            command.word(base, g.device_address());
            command.word(base + 1, k.device_address());
            command.word(base + 2, m.device_address());
            command.word(base + 3, wt.map_or(0, DeviceVector::device_address));
            command.word(base + 4, o.device_address());
        }
        let grid = [u32::try_from(rows).map_err(|_| "fast gate blend norm: grid overflow")?, count as u32, 1, fast_ops::gate_blend_norm_threads(dim) as u32, 0];
        if !self.record_resident_block(Kind::FastStreamOps, command, grid)? {
            return Err("fast gate blend norm: launches are not recorded now".into());
        }
        self.resident_completion()?;
        Ok(outputs)
    }

    /// Multi-chat decode direct KV finish (`FAST_NORM`, `NORM_MODE_KV_DIRECT`): fuses Key (`ShortConv` + `HeadClip`
    /// and `PackBf16`) and Value (`ShortConv` + `RoundedNorm` and `PackBf16`), writing packed BF16 rows directly into
    /// each chat's KV planes and writing next rings to `key_every` / `val_every` in one launch.
    #[allow(clippy::too_many_arguments, reason = "paired key/value shapes, scalars, and per-chat ring/plane targets")]
    pub fn fast_kv_finish_direct_into(
        &self, rows: usize, kv_heads: usize, taps: usize, plane_rows: usize,
        key_bound: f32, val_eps: f32, key_scale: f32, val_scale: f32,
        key_in: &DeviceVector<'_>, key_fixed: &DeviceVector<'_>, key_made: Option<&DeviceVector<'_>>, key_every: &mut DeviceVector<'_>,
        val_in: &DeviceVector<'_>, val_fixed: &DeviceVector<'_>, val_made: Option<&DeviceVector<'_>>, val_every: &mut DeviceVector<'_>,
        chats: &[(&DeviceVector<'_>, &DeviceVector<'_>, usize, &DeviceVector<'_>, &DeviceVector<'_>, usize)],
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast kv direct: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let width = kv_heads.checked_mul(fast_ops::KV_FINISH_HEAD_DIM).ok_or("fast kv direct: width overflow")?;
        let chat_spans: Vec<_> = chats.iter().map(|&(kr, vr, rs, kp, vp, ps)| (span(kr), span(vr), rs, span(kp), span(vp), ps)).collect();
        fast_ops::check_kv_finish_direct(
            rows, kv_heads, taps, plane_rows, key_bound, val_eps, key_scale, val_scale,
            span(key_in), span(key_fixed), key_made.map(span), span(key_every),
            span(val_in), span(val_fixed), val_made.map(span), span(val_every), &chat_spans,
        )?;
        for (inp, fix, mad, ev) in [(key_in, key_fixed, key_made, &*key_every), (val_in, val_fixed, val_made, &*val_every)] {
            self.check_vector(inp, rows * width)?;
            self.check_vector(fix, width * taps)?;
            self.check_vector(ev, rows * taps * width)?;
            if inp.is_segmented() || fix.is_segmented() || ev.is_segmented() || !ev.writable_from(0) {
                return Err("fast kv direct: contiguous private storage required".into());
            }
            if let Some(m) = mad {
                self.check_vector(m, rows * kv_heads * taps)?;
                if m.is_segmented() { return Err("fast kv direct: made taps segmented".into()); }
            }
        }
        let row_words = width / 2;
        let mut ring_slots = 0u64;
        for (r, &(kr, vr, rs, kp, vp, ps)) in chats.iter().enumerate() {
            for ring in [kr, vr] {
                self.check_vector(ring, taps * width)?;
                if ring.is_segmented() { return Err("fast kv direct: chat ring segmented".into()); }
            }
            let min_words = ps.checked_add(1).and_then(|n| n.checked_mul(row_words)).ok_or("fast kv direct: plane overflow")?;
            let write_word = ps.checked_mul(row_words).ok_or("fast kv direct: plane overflow")?;
            for plane in [kp, vp] {
                self.check_vector_prefix(plane, min_words)?;
                if !plane.len().is_multiple_of(row_words) || !plane.writable_from(write_word) {
                    return Err("fast kv direct: chat plane requires private aligned storage".into());
                }
            }
            ring_slots |= (rs as u64) << (4 * r);
        }
        self.require_arm_kernels(&[Kind::FastNorm.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastNorm);
        let words = [
            key_in.device_address(), val_in.device_address(), key_fixed.device_address(),
            rows as u64, kv_heads as u64, fast_ops::NORM_MODE_KV_DIRECT,
            u64::from(key_bound.to_bits()) | (u64::from(val_eps.to_bits()) << 32),
            u64::from(key_scale.to_bits()) | (u64::from(val_scale.to_bits()) << 32),
            taps as u64, val_fixed.device_address(),
            key_made.map_or(0, DeviceVector::device_address), val_made.map_or(0, DeviceVector::device_address),
            key_every.device_address(), val_every.device_address(), ring_slots, 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        let row_bytes = (row_words as u64) * 4;
        for (r, &(kr, vr, _, kp, vp, ps)) in chats.iter().enumerate() {
            command.word(16 + 4 * r, kr.device_address());
            command.word(16 + 4 * r + 1, vr.device_address());
            command.word(16 + 4 * r + 2, kp.device_address() + (ps as u64) * row_bytes);
            command.word(16 + 4 * r + 3, vp.device_address() + (ps as u64) * row_bytes);
        }
        let blocks = u32::try_from((2 * rows * kv_heads).div_ceil(4)).map_err(|_| "fast kv direct: grid overflow")?;
        let grid = [blocks, 1, 1, fast_ops::NORM_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastNorm, command, grid)? {
            return Err("fast kv direct: launches are not recorded now".into());
        }
        self.resident_completion()
    }

    /// Multi-chat decode direct `ShortConv` (`FAST_NORM`, `NORM_MODE_SHORT_CONV_DIRECT`): reads each chat's ring
    /// directly without a `row_rings` join copy and writes output + per-chat next rings in one launch.
    #[allow(clippy::too_many_arguments, reason = "shape, form, scalars, and per-chat ring operands")]
    pub fn fast_short_conv_direct_into(
        &self, rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
        inputs: &DeviceVector<'_>, token_taps: Option<&DeviceVector<'_>>, addend: Option<&DeviceVector<'_>>,
        fixed_taps: &DeviceVector<'_>, output: &mut DeviceVector<'_>, every: &mut DeviceVector<'_>,
        chat_rings: &[(&DeviceVector<'_>, usize)],
    ) -> Result<(), String> {
        if self.arm_capture() { return Err("fast short conv direct: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        let span = |v: &DeviceVector<'_>| Span { address: v.device_address(), bytes: v.len() as u64 * 4 };
        let ring_spans: Vec<_> = chat_rings.iter().map(|&(r, s)| (span(r), s)).collect();
        fast_ops::check_short_conv_direct(
            rows, heads, dim, taps, form, bound,
            span(inputs), token_taps.map(span), addend.map(span), span(fixed_taps),
            span(output), span(every), &ring_spans,
        )?;
        let width = heads * dim;
        self.check_vector(inputs, rows * width)?;
        self.check_vector(fixed_taps, width * taps)?;
        self.check_vector(output, rows * width)?;
        self.check_vector(every, rows * taps * width)?;
        if inputs.is_segmented() || fixed_taps.is_segmented()
            || output.is_segmented() || !output.writable_from(0)
            || every.is_segmented() || !every.writable_from(0)
        {
            return Err("fast short conv direct: contiguous private storage required".into());
        }
        if let Some(tt) = token_taps {
            self.check_vector(tt, rows * heads * taps)?;
            if tt.is_segmented() { return Err("fast short conv direct: token taps segmented".into()); }
        }
        if let Some(ad) = addend {
            self.check_vector(ad, rows * width)?;
            if ad.is_segmented() { return Err("fast short conv direct: addend segmented".into()); }
        }
        for &(r, _) in chat_rings {
            self.check_vector(r, taps * width)?;
            if r.is_segmented() { return Err("fast short conv direct: chat ring segmented".into()); }
        }
        self.require_arm_kernels(&[Kind::FastNorm.kernel()])?;
        self.resident_completion()?;
        let mut command = Command::new(Kind::FastNorm);
        let words = [
            inputs.device_address(), fixed_taps.device_address(), output.device_address(),
            rows as u64, dim as u64, fast_ops::NORM_MODE_SHORT_CONV_DIRECT, u64::from(bound.to_bits()),
            heads as u64, taps as u64, u64::from(form),
            token_taps.map_or(0, DeviceVector::device_address), addend.map_or(0, DeviceVector::device_address),
            every.device_address(), 0, 0, 0,
        ];
        for (i, w) in words.into_iter().enumerate() { command.word(i, w); }
        for (r, &(ring, slot)) in chat_rings.iter().enumerate() {
            command.word(16 + 2 * r, ring.device_address());
            command.word(16 + 2 * r + 1, slot as u64);
        }
        let gx = u32::try_from(width.div_ceil(fast_ops::NORM_THREADS)).map_err(|_| "fast short conv direct: grid overflow")?;
        let grid = [gx, rows as u32, 1, fast_ops::NORM_THREADS as u32, 0];
        if !self.record_resident_block(Kind::FastNorm, command, grid)? {
            return Err("fast short conv direct: launches are not recorded now".into());
        }
        self.resident_completion()
    }
}
