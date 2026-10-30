//! One prompt-pass layer stage (NextSession::execute runs one per layer, in order): the layer's
//! PLE update (layer 1), the attention HC mix, attention, the MLP HC mix and MoE. The stage ends
//! after the MoE: its combine runs here, or the next stage's attention mix takes it over
//! (`pending`, flash_next_stage_plan::defers_moe), so a span's last stage leaves a whole residual.
use super::*;
use crate::flash_next_stage_plan::defers_moe;

impl<'a, 'w> NextSession<'a, 'w> {
    /// Layer `li` of a pass that ends at layer `end`, for every row of `spans` (the pass's
    /// workspace rows). `pending`: the earlier layer whose MoE combine this stage's attention mix
    /// takes over; on return, this layer's, when the next stage takes it over.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn layer_stage(
        b: &Backend<'a>,
        work: &mut Workspace<'a>,
        sessions: &mut [&mut Self],
        spans: &[std::ops::Range<usize>],
        verify: bool,
        li: usize,
        end: usize,
        unobserved: bool,
        pending: &mut Option<usize>,
    ) -> Result<()> {
        Self::stage_mix(b, work, sessions, spans, verify, li, pending)?;
        Self::stage_body(b, work, sessions, spans, verify, li, end, unobserved, pending)
    }
    /// The first part of layer `li`'s stage: the PLE update (layer 1) and the attention HC mix
    /// into the mixed rows, which takes over `pending`'s MoE combine.
    pub(super) fn stage_mix(
        b: &Backend<'a>,
        work: &mut Workspace<'a>,
        sessions: &mut [&mut Self],
        spans: &[std::ops::Range<usize>],
        verify: bool,
        li: usize,
        pending: &mut Option<usize>,
    ) -> Result<()> {
        let count = spans.last().unwrap().end;
        // Passes above 16 rows come only from a 32-row tick (VADE_VERIFY_ROWS): stage PLE at once.
        let ple_rows = blocks::PLE_PASS_ROWS;
        if li == 1 && verify && count <= ple_rows && sessions.iter().all(|s| s.dump.is_none()) {
            // Key/value projections and gates once for the pass (row kernels keep each
            // row's one-row sums); every row then convolves in order after its snapshot.
            let base = work.residual.device_address();
            let (input, gated) = sessions[0].ple.stage_pass(b, &work.lexical, base, count)?;
            for (s, span) in sessions.iter_mut().zip(spans) {
                for row in span.clone() {
                    let saved = &mut s.saved[row - span.start];
                    s.ple.copy_history(b, &mut saved.ple)?;
                    s.ple.select_history(&mut saved.ple);
                    let at = |a: u64, width: usize| a + (row * width * 4) as u64;
                    s.ple.convolve_row(b, at(input, W), at(base, W), at(gated, W))?;
                }
            }
        } else if li == 1 {
            for (s, span) in sessions.iter_mut().zip(spans) {
                let tile = if !verify && s.dump.is_none() {
                    blocks::PLE_TOKENS
                } else {
                    1
                };
                if tile == blocks::PLE_TOKENS {
                    // Project four tiles per pass over the weights; gate and convolve
                    // the same 64-row tiles in the same order.
                    for slice in span.clone().step_by(blocks::PLE_SLICE) {
                        let slice_end = (slice + blocks::PLE_SLICE).min(span.end);
                        s.ple.project_slice(
                            b,
                            &work.lexical.slice(slice * H..slice_end * H)?,
                            slice_end - slice,
                        )?;
                        for first in (slice..slice_end).step_by(tile) {
                            let end = (first + tile).min(slice_end);
                            s.ple.finish_tile(
                                b,
                                &mut work.residual.slice(first * W..end * W)?,
                                first - slice,
                                end - first,
                            )?;
                        }
                    }
                    continue;
                }
                for first in span.clone().step_by(tile) {
                    let end = (first + tile).min(span.end);
                    let n = end - first;
                    for row in first..end {
                        s.rec("layer1.ple_e", &work.prompt.1[row * H..(row + 1) * H]);
                    }
                    let at = first * H;
                    if verify {
                        let saved = &mut s.saved[first - span.start];
                        s.ple.copy_history(b, &mut saved.ple)?;
                        s.ple.select_history(&mut saved.ple);
                    }
                    s.ple.project_next(
                        b,
                        &mut work.residual.slice(first * W..end * W)?,
                        &work.lexical.slice(at..at + n * H)?,
                        n,
                    )?;
                    if s.dump.is_some() {
                        s.rec(
                            "layer1.ple",
                            &b.read_vector_range(&work.residual, first * W, W)?,
                        );
                    }
                }
            }
        }
        if let Some(prev) = pending.take() {
            let finish = work.deferred_finish.take();
            sessions[0].attn_hc[li].combine_mix_tiled(
                &sessions[0].mlp_hc[prev],
                b,
                &mut work.hc,
                &mut work.residual,
                &work.block,
                &work.mixed,
                count,
                finish,
            )?;
        } else {
            sessions[0].attn_hc[li].mix_tiled(
                b,
                &mut work.hc,
                &work.residual,
                &work.mixed,
                count,
            )?;
        }
        Self::check_ops(
            b,
            sessions,
            spans,
            &format!("layer{li}.hc_attn"),
            &work.mixed,
            H,
        )?;
        Ok(())
    }
    /// The rest of layer `li`'s stage, from attention on the mixed rows (stage_mix) to the
    /// MoE. On return `pending` names this layer when its combine is left to the next mix.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage_body(
        b: &Backend<'a>,
        work: &mut Workspace<'a>,
        sessions: &mut [&mut Self],
        spans: &[std::ops::Range<usize>],
        verify: bool,
        li: usize,
        end: usize,
        unobserved: bool,
        pending: &mut Option<usize>,
    ) -> Result<()> {
        let model = sessions[0].model;
        let count = spans.last().unwrap().end;
        if li % 4 == 3
            && sessions
                .iter()
                .all(|s| s.indexer.select(li, s.pos, s.pos + 1) == IndexSelection::Sparse)
        {
            let mut attention = sessions
                .iter_mut()
                .map(|s| {
                    Qsa::load_next_rows(
                        b,
                        &model.weights,
                        li,
                        s.capacity,
                        model.qsa_proj[li],
                        s.qsa_state[li].take(),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let batch = work.qsa_batch.as_mut().unwrap();
            if !model.tp_qsa_project(b, li, &attention[0], batch, &work.mixed, count, attention.len())? { attention[0].project_batch(b, batch, &work.mixed, count)?; }
            for (q, span) in attention.iter_mut().zip(spans) {
                work.qsa.use_norm(model.qsa_norms[li].as_ref().unwrap())?;
                q.attend_batch_row(b, batch, span.start, &mut work.qsa, span.len())?;
            }
            if !attention[0].out_operands(batch).map_or(Ok(false), |o| model.tp_out_project(b, li, o, &mut work.block, count, attention.len()))? { attention[0].finish_batch(b, batch, &mut work.block, count)?; }
            for (s, q) in sessions.iter_mut().zip(attention) {
                s.qsa_state[li] = Some(q.into_state());
            }
        } else if li % 4 == 3 {
            // Custom dense hooks share the same projection schedule. Sparse
            // is the model/default path, and its projections are pooled above.
            for (s, span) in sessions.iter_mut().zip(spans) {
                let input = work.mixed.slice(span.start * H..span.end * H)?;
                work.qsa.use_norm(model.qsa_norms[li].as_ref().unwrap())?;
                let mut q = Qsa::load_next_rows(
                    b,
                    &model.weights,
                    li,
                    s.capacity,
                    model.qsa_proj[li],
                    s.qsa_state[li].take(),
                )?;
                q.project(b, &mut work.qsa, &input, span.len())?;
                if s.indexer.select(li, s.pos, s.pos + 1) == IndexSelection::Dense {
                    q.attend_dense(b, &mut work.qsa)?;
                } else {
                    q.attend(b, &mut work.qsa)?;
                }
                b.copy_range(
                    work.qsa.output(),
                    &mut work.block,
                    0,
                    span.start * H,
                    span.len() * H,
                )?;
                s.qsa_state[li] = Some(q.into_state());
            }
        } else {
            let mut attention = sessions
                .iter_mut()
                .map(|s| model.gdn_load(b, li, s.gdn_state[li].take()))
                .collect::<Result<Vec<_>>>()?;
            work.gdn
                .copy_norm(b, model.gdn_norms[li].as_ref().unwrap())?;
            if !model.tp_gdn_project(b, li, &attention[0], &mut work.gdn, &work.mixed, count, attention.len())? { attention[0].project_next(b, &work.gdn, &work.mixed, count)?; }
            // Every session's call fully fused (prompt rows, no verify): the
            // gated scans write the out projection's packed input directly.
            if !verify && count >= 64 && spans.iter().all(|span| gdn::fuses_fully(span.len())) {
                let tile = b.flash_dense_tile(attention[0].out_cols(), count)?;
                work.gdn.packed_out.set(Some((tile, count)));
            }
            // VADE_GDN_COHORT: a multi-chat verify pass whose windows all run the rows kernel
            // uploads one descriptor table and gates every row in one launch (same words).
            let cohort = verify && sessions.len() > 1 && sessions.len() <= COHORT_TABLE_CHATS
                && gdn::cohort_verify() && gdn::chunk_form()
                && spans.iter().all(|span| span.len() <= spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT);
            if cohort {
                let table: Vec<f32> = attention.iter().flat_map(|g| g.descriptor_words()).collect();
                work.recurrence.resize_within_capacity(table.len())?;
                b.write_vector(&mut work.recurrence, &table)?;
                work.gdn.defer_gate.set(true);
            }
            // gdn::cohort_copies: every chat's entry history and tape, before any of this layer's scans.
            let copies = cohort && gdn::cohort_copies();
            // gdn::cohort_kernels: every chat's rows and history in one launch each.
            let kernels = cohort && gdn::cohort_kernels() && gdn::cohort_kernel_entries(b);
            let scans = (|| -> Result<()> {
            if copies {
                let (mut wide, mut narrow) = (Vec::with_capacity(2 * sessions.len()), Vec::with_capacity(2 * sessions.len()));
                for ((s, g), span) in sessions.iter_mut().zip(attention.iter()).zip(spans) {
                    let rollback = s.rollback.as_mut().ok_or("MTP rollback missing")?;
                    wide.push(g.history_copy_row(rollback.gdn[li].as_ref().ok_or("MTP entry state missing")?)?);
                    let [qkv, a, gb] = rollback.inputs[li].as_mut().ok_or("MTP GDN tape missing")?.capture_rows(&work.gdn, span.start, span.len())?;
                    wide.push(qkv); narrow.extend([a, gb]);
                }
                gdn::hybrid_copy_rows(b, &mut work.copy_rows, &[&wide, &narrow])?;
                work.gdn.entry_copied.set(true);
            }
            if kernels { *work.gdn.cohort.borrow_mut() = Some(gdn::CohortLaunches::default()); }
            for (chat, ((s, g), span)) in sessions.iter_mut().zip(attention.iter_mut()).zip(spans).enumerate() {
                let pos = s.pos;
                let table = cohort.then(|| work.recurrence.device_address() + (chat * 24) as u64);
                if verify {
                    let rollback = s.rollback.as_mut().ok_or("MTP rollback missing")?;
                    rollback.shallow = gdn::shallow_verify(pos, span.len());
                    let tape = rollback.inputs[li].as_mut().ok_or("MTP GDN tape missing")?;
                    // With gdn::cohort_copies the layer's tapes and entry histories are already copied.
                    let tape = if copies { None } else if gdn::fused_verify_copy(b, &work.recurrence) {
                        Some(tape.capture_ranges(&work.gdn, span.start, span.len(), g.tape_widths())?)
                    } else {
                        tape.capture(b, &work.gdn, span.start, span.len(), g.tape_widths())?;
                        None
                    };
                    g.verify_next(
                        b,
                        &work.gdn,
                        span.start,
                        span.len(),
                        &mut work.recurrence,
                        rollback.gdn[li].as_mut().ok_or("MTP entry state missing")?,
                        pos,
                        tape,
                        table,
                    )?;
                } else {
                    g.recur_next(b, &work.gdn, span.start, span.len(), &mut work.recurrence, pos)?;
                }
            }
            if kernels { gdn::launch_cohort(b, &work.gdn, &mut work.gdn_launches)?; }
            Ok(())
            })();
            work.gdn.entry_copied.set(false);
            work.gdn.cohort.borrow_mut().take();
            if cohort {
                work.gdn.defer_gate.set(false);
                work.recurrence.resize_within_capacity(6)?;
            }
            scans?;
            if cohort {
                attention[0].gate_rows(b, &work.gdn, 0, count)?;
            }
            if !model.tp_gdn_finish(b, li, &attention[0], &work.gdn, &mut work.block, count, attention.len())? { attention[0].finish_next(b, &work.gdn, &work.block, count)?; }
            for (s, g) in sessions.iter_mut().zip(attention) {
                s.gdn_state[li] = Some(g.into_state());
            }
        }
        Self::check_ops(
            b,
            sessions,
            spans,
            &format!("layer{li}.attention"),
            &work.block,
            H,
        )?;
        for (s, span) in sessions.iter_mut().zip(spans) {
            if s.dump.is_some() {
                s.rec(
                    &format!("layer{li}.attn"),
                    &b.read_vector_range(&work.block, span.start * H, span.len() * H)?,
                );
            }
        }
        work.mixed16_ready=false;
        if !unobserved {
            sessions[0].attn_hc[li].combine_tiled(
                b,
                &mut work.hc,
                &mut work.residual,
                &work.block,
                count,
            )?;
            Self::check_ops(
                b,
                sessions,
                spans,
                &format!("layer{li}.combine_attn"),
                &work.residual,
                W,
            )?;
            sessions[0].mlp_hc[li].mix_tiled(
                b,
                &mut work.hc,
                &work.residual,
                &work.mixed,
                count,
            )?;
        } else if !verify && crate::flash_prompt_options::cached().cold8k_r2 && crate::flash_prompt_options::Options::kernel_rows(count) {
            work.mixed16_ready=sessions[0].mlp_hc[li].combine_mix_tiled_a16(
                &sessions[0].attn_hc[li],b,&mut work.hc,&mut work.residual,&work.block,&work.mixed,&work.mixed16,count)?;
        } else {
            // The attention combine and the MLP mix's norm share one residual pass.
            sessions[0].mlp_hc[li].combine_mix_tiled(
                &sessions[0].attn_hc[li],
                b,
                &mut work.hc,
                &mut work.residual,
                &work.block,
                &work.mixed,
                count,
                None,
            )?;
        }
        Self::check_ops(
            b,
            sessions,
            spans,
            &format!("layer{li}.hc_mlp"),
            &work.mixed,
            H,
        )?;
        // A combine deferred below also takes over this layer's k16_finish.
        work.defer_finish = defers_moe(li, end, unobserved, || model.tp_plain(li, count));
        work.moe_at(b, &model.moe[li], count, None, !verify, model.tp_moe(li, count))?;
        work.defer_finish = false;
        Self::check_ops(
            b,
            sessions,
            spans,
            &format!("layer{li}.moe"),
            &work.block,
            H,
        )?;
        let routing = if sessions.iter().any(|s| s.dump.is_some()) {
            work.dump_routes(b, count)?
        } else {
            Vec::new()
        };
        // Layer 1 starts with the PLE residual update and the last layer feeds the final
        // mixer: those combines run here.
        if defers_moe(li, end, unobserved, || model.tp_plain(li, count)) {
            *pending = Some(li);
        } else {
            sessions[0].mlp_hc[li].combine_tiled(
                b,
                &mut work.hc,
                &mut work.residual,
                &work.block,
                count,
            )?;
        }
        for (s, span) in sessions.iter_mut().zip(spans) {
            if let Some(layers) = &mut s.layer_bytes {
                use std::io::Write;
                let values =
                    b.read_vector_range(&work.residual, span.start * W, span.len() * W)?;
                let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(layers.join(format!("layer{li}.f32")))
                    .map_err(err)?
                    .write_all(&bytes)
                    .map_err(err)?;
            }
            if let Some(d) = &mut s.dump {
                d.push(dump::moe_line(li, &routing[span.start]));
            }
            if s.dump.is_some() {
                s.rec(
                    &format!("layer{li}.x"),
                    &b.read_vector_range(&work.residual, span.start * W, span.len() * W)?,
                );
            }
        }
        Ok(())
    }
}
